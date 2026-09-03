from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .prompt_router import VideoFlyPrompt
from .raan import RAAN, exo_triplet_loss, ranking_loss


class ContinualSystem(nn.Module):
    """The task-id-free RAAN/FlyPrompt anchor used inside final FlyGCL."""

    def __init__(self, config: Dict):
        super().__init__()
        if config.get("benchmark") != "skill":
            raise ValueError("This release retains only the skill-assessment anchor")
        if config.get("view") != "ego_exo" or config.get("skill_head") != "tl":
            raise ValueError("The FlyGCL anchor requires view=ego_exo and skill_head=tl")
        if config.get("method") != "flyprompt":
            raise ValueError("Only the FlyGCL prompt/router anchor is included")
        self.config = config
        model_cfg = config.get("model", {})
        input_dim = int(model_cfg.get("input_dim", 1024))
        self.raan = RAAN(
            output_dim=1,
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("attention_hidden_dim", 256)),
            num_filters=int(model_cfg.get("num_filters", 3)),
        )
        prompt_cfg = config.get("flyprompt", {})
        self.flyprompt = VideoFlyPrompt(
            num_tasks=4,
            num_tokens=int(model_cfg.get("num_tokens", 10)),
            input_dim=input_dim,
            expansion_dim=int(prompt_cfg.get("expansion_dim", 5000)),
            ridge=float(prompt_cfg.get("ridge", 1e4)),
            init_std=float(prompt_cfg.get("init_std", 0.05)),
            init_gate=float(prompt_cfg.get("init_gate", 0.2)),
            normalize_router=bool(prompt_cfg.get("normalize_router", False)),
            prototype_weight=float(prompt_cfg.get("prototype_weight", 0.0)),
            router_temperature=float(prompt_cfg.get("router_temperature", 1.0)),
        )

    def begin_task(self, task_id: int) -> None:
        self.flyprompt.begin_task(task_id)

    @staticmethod
    def router_query(batch: Dict[str, object]) -> torch.Tensor:
        return 0.5 * (
            batch["better"].mean(dim=1) + batch["worse"].mean(dim=1)  # type: ignore[union-attr]
        )

    def forward_skill(self, batch: Dict[str, object], training: bool) -> Dict[str, torch.Tensor]:
        task_ids = (
            batch["task_id"].long()  # type: ignore[union-attr]
            if training
            else self.flyprompt.predict_task(self.router_query(batch))
        )
        better = self.flyprompt.apply(batch["better"], task_ids)  # type: ignore[arg-type]
        worse = self.flyprompt.apply(batch["worse"], task_ids)  # type: ignore[arg-type]
        exo = self.flyprompt.apply(batch["exo"], task_ids)  # type: ignore[arg-type]
        better_result = self.raan(better)
        worse_result = self.raan(worse)
        return {
            "score_better": better_result["output"].squeeze(-1),
            "score_worse": worse_result["output"].squeeze(-1),
            "better_embedding": better_result["embedding"],
            "worse_embedding": worse_result["embedding"],
            "exo_embedding": self.raan(exo)["embedding"],
            "routed_task": task_ids,
        }

    def loss(self, batch: Dict[str, object]) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = self.forward_skill(batch, training=True)
        cfg = self.config.get("loss", {})
        rank = ranking_loss(
            result["score_better"], result["score_worse"], float(cfg.get("margin", 1.0))
        )
        triplet = exo_triplet_loss(
            result["better_embedding"],
            result["worse_embedding"],
            result["exo_embedding"],
            cfg.get("triplet_margin"),
        )
        total = rank + float(cfg.get("triplet_weight", 0.1)) * triplet
        return total, {"ranking": float(rank.detach()), "triplet": float(triplet.detach())}

    @torch.no_grad()
    def collect_router(self, batch: Dict[str, object], task_id: int) -> None:
        query = self.router_query(batch)
        labels = torch.full(
            (query.shape[0],), int(task_id), dtype=torch.long, device=query.device
        )
        self.flyprompt.router.collect(query, labels)
        self.flyprompt.collect_prototypes(query, labels)

    @torch.no_grad()
    def update_router(self) -> None:
        self.flyprompt.router.update()
