from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from flygcl.skill.ego_model import EgoTemporalMoE


class RNTemporalMoE(nn.Module):
    """RN head: score each ego/exo pair after 20-token concatenation.

    Multiple action-matched exo references are integrated with a label-free
    cosine gate.  The underlying four-scale temporal MoE is initialized from
    the corresponding frozen Ego-only expert, then fine-tuned only as an
    Ego-exo residual expert.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        ego_tokens: int = 10,
        dropout: float = 0.12,
        reference_temperature: float = 0.35,
        relation_weight: float = 0.35,
        pair_correction_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.reference_temperature = max(float(reference_temperature), 1e-3)
        self.relation_weight = float(relation_weight)
        self.pair_correction_weight = float(pair_correction_weight)
        self.core = EgoTemporalMoE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            tokens=2 * int(ego_tokens),
            dropout=dropout,
        )
        self.relation_input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.relation_score = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_correction = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @torch.no_grad()
    def initialize_from_ego(self, state: Mapping[str, torch.Tensor]) -> None:
        target = self.core.state_dict()
        for key, value in state.items():
            if key == "position":
                target[key][:, : value.shape[1]].copy_(value)
                target[key][:, value.shape[1] :].copy_(value)
            elif key in target and target[key].shape == value.shape:
                target[key].copy_(value)
        self.core.load_state_dict(target)

    def _score(self, ego: torch.Tensor, exo: torch.Tensor) -> Dict[str, torch.Tensor]:
        if exo.ndim == 3:
            exo = exo.unsqueeze(1)
        batch, references, tokens, dimension = exo.shape
        ego_repeated = ego[:, None].expand(-1, references, -1, -1)
        sequence = torch.cat((ego_repeated, exo), dim=2).reshape(
            batch * references, ego.shape[1] + tokens, dimension
        )
        result = self.core.forward_video(sequence)
        ego_descriptor = F.normalize(ego.mean(1), dim=-1)
        exo_descriptor = F.normalize(exo.mean(2), dim=-1)
        similarity = torch.einsum("bd,brd->br", ego_descriptor, exo_descriptor)
        weights = torch.softmax(similarity / self.reference_temperature, dim=1)
        scores = result["score"].reshape(batch, references)
        expert_scores = result["expert_scores"].reshape(batch, references, -1)
        ego_relation = self.relation_input(ego.mean(1))[:, None].expand(-1, references, -1)
        exo_relation = self.relation_input(exo.mean(2))
        relation = torch.cat(
            (
                ego_relation,
                exo_relation,
                (ego_relation - exo_relation).abs(),
                ego_relation * exo_relation,
            ),
            dim=-1,
        )
        relation_scores = self.relation_score(relation).squeeze(-1)
        return {
            "score": (weights * (scores + self.relation_weight * relation_scores)).sum(1),
            "expert_scores": (weights[..., None] * expert_scores).sum(1),
            "reference_entropy": -(
                weights * torch.log(weights.clamp_min(1e-8))
            ).sum(1),
        }

    def forward_pair(
        self, better: torch.Tensor, worse: torch.Tensor, exo: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        positive = self._score(better, exo)
        negative = self._score(worse, exo)
        expert_margins = positive["expert_scores"] - negative["expert_scores"]
        correction = 0.5 * (
            self.pair_correction(expert_margins)
            - self.pair_correction(-expert_margins)
        ).squeeze(-1)
        base_margin = positive["score"] - negative["score"]
        return {
            "margin": base_margin + self.pair_correction_weight * correction,
            "base_margin": base_margin,
            "pair_correction": correction,
            "score_better": positive["score"],
            "score_worse": negative["score"],
            "expert_margins": expert_margins,
            "gate": torch.softmax(
                0.5 * (positive["expert_scores"] + negative["expert_scores"]), dim=-1
            ),
            "reference_entropy": 0.5
            * (positive["reference_entropy"] + negative["reference_entropy"]),
        }
