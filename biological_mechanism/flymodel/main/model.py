"""Independent fixed encoder and temporal expert bank."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .config import Config


class FixedEncoder(nn.Module):
    def __init__(self, orn_pn: np.ndarray, pn_kc: np.ndarray, cfg: Config, device: str):
        super().__init__(); self.cfg = cfg
        if orn_pn.shape != (cfg.n_pn, cfg.orn_per_channel) or orn_pn.dtype != np.float32 or not np.allclose(orn_pn.sum(1), 1, atol=1e-6):
            raise ValueError("invalid ORN-PN asset")
        if pn_kc.shape != (cfg.n_pn, cfg.n_kc) or pn_kc.dtype != np.float32 or np.any(pn_kc < 0) or np.any(np.count_nonzero(pn_kc, axis=0) == 0):
            raise ValueError("invalid PN-KC asset")
        self.register_buffer("orn_pn", torch.from_numpy(np.array(orn_pn, copy=True)))
        self.register_buffer("pn_kc", torch.from_numpy(np.array(pn_kc, copy=True)))
        self.to(device)

    def forward(self, odors: torch.Tensor, sigma: float = 0, generator: torch.Generator | None = None) -> torch.Tensor:
        orn = odors.unsqueeze(-1).expand(-1, -1, self.cfg.orn_per_channel)
        if sigma:
            orn = orn + sigma * torch.randn(orn.shape, device=orn.device, dtype=orn.dtype, generator=generator)
        pn = F.relu(torch.sum(orn * self.orn_pn.unsqueeze(0), dim=-1))
        kc = F.relu(pn @ self.pn_kc)
        values, indices = kc.topk(self.cfg.kc_topk, dim=-1)
        return torch.zeros_like(kc).scatter(-1, indices, values)


class ExpertBank(nn.Module):
    def __init__(self, seed: int, n_experts: int, n_timescales: int, cfg: Config, device: str):
        super().__init__(); self.n_experts = n_experts; self.n_timescales = n_timescales
        generator = torch.Generator().manual_seed(seed + 510_000)
        self.heads = nn.ModuleList()
        for _ in range(n_experts * n_timescales):
            head = nn.Linear(cfg.n_kc, cfg.n_classes, bias=False)
            nn.init.normal_(head.weight, std=cfg.n_kc ** -0.5, generator=generator)
            self.heads.append(head)
        self.to(device)

    def head(self, expert: int, timescale: int) -> nn.Linear:
        return self.heads[expert * self.n_timescales + timescale]

    def expert_logits(self, kc: torch.Tensor, expert: int) -> torch.Tensor:
        return torch.stack([self.head(expert, scale)(kc) for scale in range(self.n_timescales)], dim=1)

    def all_logits(self, kc: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.expert_logits(kc, expert) for expert in range(self.n_experts)], dim=1)
