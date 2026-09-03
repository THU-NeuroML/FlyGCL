from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RPFC(nn.Module):
    """Random-projection, closed-form task classifier used by FlyPrompt."""

    def __init__(self, input_dim: int, expansion_dim: int, num_tasks: int, ridge: float, normalize: bool = False):
        super().__init__()
        self.input_dim = int(input_dim)
        self.expansion_dim = int(expansion_dim)
        self.num_tasks = int(num_tasks)
        self.ridge = float(ridge)
        self.normalize = bool(normalize)
        self.register_buffer("random_weight", torch.randn(self.input_dim, self.expansion_dim) / self.input_dim**0.5)
        self.register_buffer("gram", torch.zeros(self.expansion_dim, self.expansion_dim))
        self.register_buffer("targets", torch.zeros(self.expansion_dim, self.num_tasks))
        self.register_buffer("weight", torch.zeros(self.num_tasks, self.expansion_dim))
        self.register_buffer("seen_tasks", torch.zeros(self.num_tasks, dtype=torch.bool))

    def expand(self, features: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            features = F.normalize(features, dim=-1)
        return F.relu(features @ self.random_weight)

    @torch.no_grad()
    def collect(self, features: torch.Tensor, task_ids: torch.Tensor) -> None:
        expanded = self.expand(features)
        one_hot = F.one_hot(task_ids.long(), num_classes=self.num_tasks).to(expanded.dtype)
        self.gram.add_(expanded.T @ expanded)
        self.targets.add_(expanded.T @ one_hot)
        self.seen_tasks[task_ids.long().unique()] = True

    @torch.no_grad()
    def update(self) -> None:
        regularized = self.gram + self.ridge * torch.eye(
            self.expansion_dim, device=self.gram.device, dtype=self.gram.dtype
        )
        try:
            solution = torch.linalg.solve(regularized, self.targets)
        except RuntimeError:
            solution = torch.linalg.lstsq(regularized, self.targets).solution
        self.weight.copy_(solution.T)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.expand(features) @ self.weight.T
        return logits.masked_fill(~self.seen_tasks.view(1, -1), torch.finfo(logits.dtype).min)


class VideoFlyPrompt(nn.Module):
    def __init__(
        self,
        num_tasks: int = 4,
        num_tokens: int = 10,
        input_dim: int = 1024,
        expansion_dim: int = 5000,
        ridge: float = 1e4,
        init_std: float = 0.05,
        init_gate: float = 0.2,
        normalize_router: bool = False,
        prototype_weight: float = 0.0,
        router_temperature: float = 1.0,
    ):
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.num_tokens = int(num_tokens)
        self.input_dim = int(input_dim)
        self.prompts = nn.Parameter(torch.empty(self.num_tasks, self.num_tokens, self.input_dim))
        nn.init.trunc_normal_(self.prompts, std=float(init_std))
        gate = min(max(float(init_gate), 1e-4), 1.0 - 1e-4)
        self.prompt_gates = nn.Parameter(torch.full((self.num_tasks,), torch.logit(torch.tensor(gate)).item()))
        self.router = RPFC(input_dim, expansion_dim, num_tasks, ridge, normalize_router)
        self.prototype_weight = float(prototype_weight)
        self.router_temperature = max(float(router_temperature), 1e-6)
        self.register_buffer("prototype_sums", torch.zeros(self.num_tasks, self.input_dim))
        self.register_buffer("prototype_counts", torch.zeros(self.num_tasks))
        self.register_buffer("trainable_task_mask", torch.zeros(self.num_tasks))
        self.current_task = 0
        self._frozen_prompts: Optional[torch.Tensor] = None
        self._frozen_gates: Optional[torch.Tensor] = None
        self.prompts.register_hook(self._mask_prompt_grad)
        self.prompt_gates.register_hook(self._mask_gate_grad)

    def _mask_prompt_grad(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self.trainable_task_mask.view(-1, 1, 1)

    def _mask_gate_grad(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self.trainable_task_mask

    @torch.no_grad()
    def begin_task(self, task_id: int) -> None:
        task_id = int(task_id)
        if task_id > 0:
            self.prompts[task_id].copy_(self.prompts[:task_id].mean(dim=0))
            self.prompt_gates[task_id].copy_(self.prompt_gates[:task_id].mean())
        self.current_task = task_id
        self.trainable_task_mask.zero_()
        self.trainable_task_mask[task_id] = 1.0
        self._frozen_prompts = self.prompts[:task_id].detach().clone()
        self._frozen_gates = self.prompt_gates[:task_id].detach().clone()

    @torch.no_grad()
    def enforce_frozen(self) -> None:
        """Undo optimizer momentum/weight-decay changes to prompts from older tasks."""
        task_id = int(self.current_task)
        if task_id > 0 and self._frozen_prompts is not None and self._frozen_gates is not None:
            self.prompts[:task_id].copy_(self._frozen_prompts)
            self.prompt_gates[:task_id].copy_(self._frozen_gates)

    @staticmethod
    def pooled(features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=1) if features.ndim == 3 else features

    def route_logits(self, query: torch.Tensor) -> torch.Tensor:
        """Return masked task scores suitable for hard or soft routing."""
        pooled = self.pooled(query)
        rpfc_logits = self.router(pooled)
        if self.prototype_weight <= 0.0 or int(self.router.seen_tasks.sum()) <= 1:
            return rpfc_logits / self.router_temperature
        prototypes = self.prototype_sums / self.prototype_counts.clamp_min(1.0).unsqueeze(1)
        cosine = F.normalize(pooled, dim=-1) @ F.normalize(prototypes, dim=-1).T
        unseen = ~self.router.seen_tasks.view(1, -1)
        cosine = cosine.masked_fill(unseen, torch.finfo(cosine.dtype).min)
        # Normalize both score families over seen tasks before mixing them.
        seen = self.router.seen_tasks
        rpfc_seen = rpfc_logits[:, seen]
        cosine_seen = cosine[:, seen]
        rpfc_norm = (rpfc_seen - rpfc_seen.mean(1, keepdim=True)) / rpfc_seen.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
        cosine_norm = (cosine_seen - cosine_seen.mean(1, keepdim=True)) / cosine_seen.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
        mixed = (1.0 - self.prototype_weight) * rpfc_norm + self.prototype_weight * cosine_norm
        mixed = mixed / self.router_temperature
        scores = pooled.new_full((pooled.shape[0], self.num_tasks), torch.finfo(pooled.dtype).min)
        scores[:, seen] = mixed
        return scores

    def predict_task(self, query: torch.Tensor) -> torch.Tensor:
        return self.route_logits(query).argmax(dim=1)

    @torch.no_grad()
    def collect_prototypes(self, features: torch.Tensor, task_ids: torch.Tensor) -> None:
        pooled = self.pooled(features)
        for task_id in task_ids.long().unique():
            selected = pooled[task_ids.long() == task_id]
            self.prototype_sums[task_id].add_(selected.sum(dim=0))
            self.prototype_counts[task_id].add_(selected.shape[0])

    def apply(self, features: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        task_ids = task_ids.long().to(features.device).clamp(0, self.num_tasks - 1)
        prompts = self.prompts[task_ids]
        if prompts.shape[1] != features.shape[1]:
            prompts = F.interpolate(
                prompts.transpose(1, 2), size=features.shape[1], mode="linear", align_corners=False
            ).transpose(1, 2)
        gates = torch.sigmoid(self.prompt_gates[task_ids]).view(-1, 1, 1)
        return features + gates * prompts

    def forward(
        self,
        features: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if task_ids is None:
            task_ids = self.predict_task(features if query is None else query)
        return self.apply(features, task_ids)
