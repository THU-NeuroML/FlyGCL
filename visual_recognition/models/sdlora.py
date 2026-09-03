import logging
import math

import timm
import torch
import torch.nn as nn

import models.vit as vit

logger = logging.getLogger()


class LoRAQKVAdapter(nn.Module):
    """Wrap the ViT Attention.qkv linear with multi-task SD-LoRA style adapters.

    This module keeps the original qkv projection frozen and adds low-rank
    adapters on Q and V for each task. Directions (A,B) are per-task, while
    magnitudes (alpha) are trainable across tasks.
    """

    def __init__(self, qkv: nn.Linear, dim: int, rank: int, alpha: float, n_tasks: int):
        super().__init__()
        self.base_qkv = qkv
        self.dim = dim
        self.rank = rank
        self.n_tasks = n_tasks
        self.current_task = 0

        # Per-task low-rank directions for Q and V
        self.A_q = nn.ModuleList([nn.Linear(dim, rank, bias=False) for _ in range(n_tasks)])
        self.B_q = nn.ModuleList([nn.Linear(rank, dim, bias=False) for _ in range(n_tasks)])
        self.A_v = nn.ModuleList([nn.Linear(dim, rank, bias=False) for _ in range(n_tasks)])
        self.B_v = nn.ModuleList([nn.Linear(rank, dim, bias=False) for _ in range(n_tasks)])

        # Per-task magnitudes (scaling factors) for Q and V
        self.alpha_q = nn.ParameterList([nn.Parameter(torch.ones(1) * alpha) for _ in range(n_tasks)])
        self.alpha_v = nn.ParameterList([nn.Parameter(torch.ones(1) * alpha) for _ in range(n_tasks)])

        # Initialize directions following original SD-LoRA: A ~ Kaiming, B ~ 0
        for ml in [self.A_q, self.A_v]:
            for m in ml:
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        for ml in [self.B_q, self.B_v]:
            for m in ml:
                nn.init.zeros_(m.weight)

        # Freeze the base qkv projection
        for p in self.base_qkv.parameters():
            p.requires_grad = False

    def set_active_task(self, task_id: int) -> None:
        """Set which task's directions are considered the "current" one.

        This affects the orthogonal loss computation; forward always uses
        all tasks up to current_task to build the final Q/V.
        """
        self.current_task = min(max(task_id, 0), self.n_tasks - 1)

    def set_task_trainable(self, active_task: int) -> None:
        """Enable gradients only for the active task's LoRA parameters."""
        for t in range(self.n_tasks):
            requires_grad = t == active_task
            modules = (self.A_q[t], self.B_q[t], self.A_v[t], self.B_v[t])
            for m in modules:
                for p in m.parameters():
                    p.requires_grad = requires_grad
            self.alpha_q[t].requires_grad = requires_grad
            self.alpha_v[t].requires_grad = requires_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, dim]
        base_out = self.base_qkv(x)  # [B, N, 3*dim]
        B, N, _ = base_out.shape
        base_q, base_k, base_v = base_out.chunk(3, dim=-1)

        x_flat = x.reshape(-1, self.dim)
        q_delta = torch.zeros_like(x_flat)
        v_delta = torch.zeros_like(x_flat)

        # Previous tasks: normalized directions + per-task scaling
        for t in range(self.current_task):
            Aq, Bq = self.A_q[t], self.B_q[t]
            Av, Bv = self.A_v[t], self.B_v[t]

            q_dir = Bq(Aq(x_flat))
            q_norm = (Aq.weight.norm(p="fro") * Bq.weight.norm(p="fro") + 1e-6)
            q_delta = q_delta + self.alpha_q[t] * (q_dir / q_norm)

            v_dir = Bv(Av(x_flat))
            v_norm = (Av.weight.norm(p="fro") * Bv.weight.norm(p="fro") + 1e-6)
            v_delta = v_delta + self.alpha_v[t] * (v_dir / v_norm)

        # Current task: un-normalized direction + scaling (as in original SD-LoRA)
        if self.current_task < self.n_tasks:
            t = self.current_task
            Aq, Bq = self.A_q[t], self.B_q[t]
            Av, Bv = self.A_v[t], self.B_v[t]

            q_delta = q_delta + self.alpha_q[t] * Bq(Aq(x_flat))
            v_delta = v_delta + self.alpha_v[t] * Bv(Av(x_flat))

        q_delta = q_delta.view(B, N, self.dim)
        v_delta = v_delta.view(B, N, self.dim)

        q = base_q + q_delta
        v = base_v + v_delta
        return torch.cat([q, base_k, v], dim=-1)

    @staticmethod
    def _cos2(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        inner = (w1 * w2).sum()
        denom = w1.norm(p="fro") * w2.norm(p="fro") + 1e-6
        return (inner / denom) ** 2

    def orthogonal_loss(self) -> torch.Tensor:
        """Compute squared cosine similarity between current and previous tasks' A."""
        t = self.current_task
        device = self.A_q[0].weight.device
        if t == 0:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)
        A_cur_q = self.A_q[t].weight
        A_cur_v = self.A_v[t].weight
        for prev in range(t):
            A_prev_q = self.A_q[prev].weight
            A_prev_v = self.A_v[prev].weight
            loss = loss + self._cos2(A_cur_q, A_prev_q) + self._cos2(A_cur_v, A_prev_v)
        return loss


class SDLoRAModel(nn.Module):
    """SD-LoRA model wrapper for FlyGCL.

    This wraps the local ViT backbone (models.vit) and injects LoRA adapters
    into selected attention blocks, keeping the backbone frozen while training
    only LoRA parameters and the classifier head.
    """

    def __init__(
        self,
        task_num: int = 10,
        num_classes: int = 100,
        backbone_name: str = None,
        sdlora_rank: int = 4,
        sdlora_alpha: float = 16.0,
        sdlora_layers: str = "all",
        sdlora_ortho_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.task_num = task_num
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.sdlora_rank = sdlora_rank
        self.sdlora_alpha = sdlora_alpha
        self.sdlora_layers = sdlora_layers
        self.sdlora_ortho_weight = sdlora_ortho_weight
        self.kwargs = kwargs

        self.task_count = 0

        assert backbone_name is not None, "backbone_name must be specified"
        # Use custom ViT model from models.vit to support local .npz loading
        if hasattr(vit, backbone_name):
            logger.info(f"Using custom ViT model: {backbone_name}")
            self.add_module("backbone", getattr(vit, backbone_name)(pretrained=True, num_classes=num_classes))
        else:
            logger.info(f"Using timm model: {backbone_name}")
            self.add_module("backbone", timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))

        # Freeze backbone parameters, keep classifier head trainable
        for _, p in self.backbone.named_parameters():
            p.requires_grad = False
        if hasattr(self.backbone, "fc"):
            self.backbone.fc.weight.requires_grad = True
            if getattr(self.backbone.fc, "bias", None) is not None:
                self.backbone.fc.bias.requires_grad = True

        self.lora_layers = []
        self._inject_lora()

    def _select_block_indices(self, depth: int):
        if self.sdlora_layers == "all":
            return list(range(depth))
        if self.sdlora_layers.startswith("last"):
            try:
                k = int(self.sdlora_layers[4:])
            except Exception:
                return list(range(depth))
            return list(range(max(0, depth - k), depth))
        try:
            idxs = [int(x) for x in self.sdlora_layers.split(",")]
            return [i for i in idxs if 0 <= i < depth]
        except Exception:
            return list(range(depth))

    def _inject_lora(self) -> None:
        dim = self.backbone.embed_dim
        depth = len(self.backbone.blocks)
        target_indices = self._select_block_indices(depth)
        logger.info(f"Injecting SD-LoRA into blocks: {target_indices}")
        for i, block in enumerate(self.backbone.blocks):
            if i in target_indices:
                attn = block.attn
                adapter = LoRAQKVAdapter(attn.qkv, dim, self.sdlora_rank, self.sdlora_alpha, self.task_num)
                attn.qkv = adapter
                self.lora_layers.append(adapter)

    def _set_active_task(self) -> None:
        for lora in self.lora_layers:
            lora.set_active_task(self.task_count)
            lora.set_task_trainable(self.task_count)

    def compute_ortho_loss(self) -> torch.Tensor:
        if self.task_count == 0 or len(self.lora_layers) == 0:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for lora in self.lora_layers:
            loss = loss + lora.orthogonal_loss()
        return loss

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        self._set_active_task()
        return self.backbone(inputs)

    def process_task_count(self) -> None:
        if self.task_count + 1 < self.task_num:
            self.task_count += 1
        logger.info(f"[SDLoRA] Switched to task {self.task_count}")
        return

