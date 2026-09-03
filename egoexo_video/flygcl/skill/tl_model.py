from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCrossViewExpert(nn.Module):
    """A task expert that scores an ego clip relative to exo demonstrations.

    The same scorer is applied to both sides of a skill pair.  Consequently the
    resulting pair margin remains antisymmetric, while the exo clips can change
    the absolute score assigned to either ego clip.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 192,
        tokens: int = 10,
        dropout: float = 0.15,
        cross_weight: float = 0.65,
        reference_temperature: float = 0.35,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cross_weight = float(cross_weight)
        self.reference_temperature = max(float(reference_temperature), 1e-3)
        self.input = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), self.hidden_dim),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, int(tokens), self.hidden_dim))
        self.local = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            padding=1,
            groups=self.hidden_dim,
        )
        self.long = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=self.hidden_dim,
        )
        self.mix = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.token_norm = nn.LayerNorm(self.hidden_dim)
        self.pool_attention = nn.Linear(self.hidden_dim, 1)
        self.summary = nn.Sequential(
            nn.LayerNorm(4 * self.hidden_dim),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.quality = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        self.cross = nn.Sequential(
            nn.LayerNorm(5 * self.hidden_dim),
            nn.Linear(5 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.input(features) + self.position[:, : features.shape[1]]
        channel_first = tokens.transpose(1, 2)
        tokens = tokens + 0.5 * self.local(channel_first).transpose(1, 2)
        tokens = tokens + 0.5 * self.long(channel_first).transpose(1, 2)
        tokens = self.token_norm(tokens + self.mix(tokens))
        weights = torch.softmax(self.pool_attention(tokens).squeeze(-1), dim=1)
        attended = (weights[..., None] * tokens).sum(1)
        statistics = torch.cat(
            (
                attended,
                tokens.mean(1),
                tokens.std(1, unbiased=False),
                tokens[:, -1] - tokens[:, 0],
            ),
            dim=-1,
        )
        return tokens, self.summary(statistics)

    def _score_encoded(
        self,
        ego_tokens: torch.Tensor,
        ego_summary: torch.Tensor,
        exo_tokens: torch.Tensor,
        exo_summary: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, references = exo_tokens.shape[:2]
        query = ego_tokens[:, None].expand(-1, references, -1, -1)
        attention = torch.einsum("brtd,brsd->brts", query, exo_tokens)
        attention = torch.softmax(attention / math.sqrt(self.hidden_dim), dim=-1)
        attended_exo = torch.einsum("brts,brsd->brtd", attention, exo_tokens).mean(2)
        ego_refs = ego_summary[:, None].expand_as(exo_summary)
        relation = torch.cat(
            (
                ego_refs,
                exo_summary,
                (ego_refs - exo_summary).abs(),
                ego_refs * exo_summary,
                attended_exo,
            ),
            dim=-1,
        )
        cross_per_reference = self.cross(relation).squeeze(-1)
        similarity = F.cosine_similarity(ego_refs, exo_summary, dim=-1)
        reference_weights = torch.softmax(
            similarity / self.reference_temperature, dim=1
        )
        cross_score = (reference_weights * cross_per_reference).sum(1)
        quality_score = self.quality(ego_summary).squeeze(-1)
        return {
            "score": quality_score + self.cross_weight * cross_score,
            "quality": quality_score,
            "cross": cross_score,
            "summary": ego_summary,
            "exo_summary": (reference_weights[..., None] * exo_summary).sum(1),
        }

    def score(self, ego: torch.Tensor, exo: torch.Tensor) -> Dict[str, torch.Tensor]:
        if exo.ndim == 3:
            exo = exo.unsqueeze(1)
        batch, references, tokens, dimension = exo.shape
        ego_tokens, ego_summary = self.encode(ego)
        exo_tokens, exo_summary = self.encode(
            exo.reshape(batch * references, tokens, dimension)
        )
        exo_tokens = exo_tokens.reshape(batch, references, tokens, self.hidden_dim)
        exo_summary = exo_summary.reshape(batch, references, self.hidden_dim)
        return self._score_encoded(ego_tokens, ego_summary, exo_tokens, exo_summary)

    def forward_pair(
        self, better: torch.Tensor, worse: torch.Tensor, exo: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if exo.ndim == 3:
            exo = exo.unsqueeze(1)
        batch, references, tokens, dimension = exo.shape
        exo_tokens, exo_summary = self.encode(
            exo.reshape(batch * references, tokens, dimension)
        )
        exo_tokens = exo_tokens.reshape(batch, references, tokens, self.hidden_dim)
        exo_summary = exo_summary.reshape(batch, references, self.hidden_dim)
        better_tokens, better_summary = self.encode(better)
        worse_tokens, worse_summary = self.encode(worse)
        positive = self._score_encoded(
            better_tokens, better_summary, exo_tokens, exo_summary
        )
        negative = self._score_encoded(
            worse_tokens, worse_summary, exo_tokens, exo_summary
        )
        return {
            "margin": positive["score"] - negative["score"],
            "quality_margin": positive["quality"] - negative["quality"],
            "cross_margin": positive["cross"] - negative["cross"],
            "quality_better": positive["quality"],
            "quality_worse": negative["quality"],
            "better_summary": positive["summary"],
            "worse_summary": negative["summary"],
            "exo_summary": 0.5 * (positive["exo_summary"] + negative["exo_summary"]),
        }


def ranking_objective(
    result: Dict[str, torch.Tensor],
    better_target: torch.Tensor,
    worse_target: torch.Tensor,
    margin: float = 0.3,
) -> tuple[torch.Tensor, Dict[str, float]]:
    def rank(value: torch.Tensor) -> torch.Tensor:
        return F.relu(float(margin) - value).mean() + F.softplus(-value / 0.5).mean()

    primary = rank(result["margin"])
    quality = rank(result["quality_margin"])
    cross = rank(result["cross_margin"])
    regression = 0.5 * (
        F.smooth_l1_loss(result["quality_better"], better_target)
        + F.smooth_l1_loss(result["quality_worse"], worse_target)
    )
    positive_distance = 1.0 - F.cosine_similarity(
        result["better_summary"], result["exo_summary"], dim=-1
    )
    negative_distance = 1.0 - F.cosine_similarity(
        result["worse_summary"], result["exo_summary"], dim=-1
    )
    alignment = F.relu(positive_distance - negative_distance + 0.1).mean()
    total = primary + 0.25 * quality + 0.30 * cross + 0.20 * regression + 0.10 * alignment
    details = {
        "loss": float(total.detach()),
        "primary": float(primary.detach()),
        "quality": float(quality.detach()),
        "cross": float(cross.detach()),
        "regression": float(regression.detach()),
        "alignment": float(alignment.detach()),
    }
    return total, details
