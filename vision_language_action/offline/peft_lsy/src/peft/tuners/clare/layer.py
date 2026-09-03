# peft/tuners/our_adapter/layer.py

from __future__ import annotations
import copy
from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, functional_call, stack_module_state
import einops
from .config import CLAREConfig, CLAREModuleConfig
from .discriminator import Discriminator, BatchedAutoEncoderSmall, get_discriminaor_class
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from .lora_layer import LoRALinear, LoRAMultiheadAttention
from .func_adapter import FuncAdapter, LoRAFuncAdapter

STACK_FORWARD = False

class ConvHelper(nn.Module):
    """Swap dims: (B, T, D) <-> (B, D, T)."""
    def forward(self, x):
        return x.transpose(1, 2)


class ResidualEnsembleGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# class FuncAdapterWrapper(nn.Module):
#     def __init__(self, 
#                  config: CLAREConfig, 
#                  adapter: nn.Module):
#         super().__init__()

#         self.add_zero_init_conv_layer = config.add_zero_init_conv_layer
#         self.func_adapter = None  # Will be set below

#         if config.add_zero_init_conv_layer:

#             conv_layer = nn.Conv1d(
#                 in_channels=config.out_feature_dim, 
#                 out_channels=config.out_feature_dim,
#                 kernel_size=1,
#                 padding=0
#             )

#             # Initialize weights and bias to zero
#             nn.init.constant_(conv_layer.weight, 0.0)
#             if conv_layer.bias is not None:
#                 nn.init.constant_(conv_layer.bias, 0.0)

#             self.func_adapter = nn.Sequential(
#                 adapter,
#                 ConvHelper(),
#                 conv_layer,
#                 ConvHelper()
#             )
#         else:
#             self.func_adapter = adapter

#     def forward(self, x):
#         if x.ndim == 2 and self.add_zero_init_conv_layer:
#             x = x.squeeze(0)
#             y = self.func_adapter(x)
#             y = y.unsqueeze(0)
#             return y
#         else:
#             return self.func_adapter(x)

def general_set_module(base_layer: nn.Module, submodule_name: str, new_submodule: nn.Module):
    if submodule_name == '':
        return "self", new_submodule
    else:
        base_layer.set_submodule(submodule_name, new_submodule)
        return submodule_name, base_layer


def general_get_module(base_layer: nn.Module, submodule_name: str):
    if submodule_name == 'self':
        return base_layer
    else:
        return base_layer.get_submodule(submodule_name)


class LoRAFuncAdapterWrapper(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lora_module = nn.Linear(input_size, output_size, bias=False)
        self.register_buffer("task_id", torch.tensor(-1, dtype=torch.int64))

    def forward(self, x):
        return self.lora_module(x)


class RPHead(nn.Module):
    """Closed-form random projection gate for adapter routing."""

    def __init__(
        self,
        feature_dim: int,
        rp_dim: int = 10000,
        ridge: float = 1e4,
        max_classes: int = 64,
    ) -> None:
        super().__init__()
        if feature_dim is None or feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if max_classes <= 0:
            raise ValueError(f"max_classes must be positive, got {max_classes}")

        self.feature_dim = feature_dim
        self.ridge = ridge
        self.max_classes = max_classes
        self.use_rp = rp_dim is not None and rp_dim > 0
        self.proj_dim = rp_dim if self.use_rp else feature_dim

        if self.use_rp:
            self.register_buffer("W_rand", torch.randn(feature_dim, self.proj_dim))
        else:
            self.register_buffer("W_rand", torch.empty(0))

        self.register_buffer("G", torch.zeros(self.proj_dim, self.proj_dim))
        self.register_buffer("Q", torch.zeros(self.proj_dim, self.max_classes))
        self.register_buffer("weight", torch.zeros(self.max_classes, self.proj_dim))
        self.register_buffer("num_classes_seen", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("needs_update", torch.tensor(False, dtype=torch.bool))

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x
        if x.ndim < 2:
            raise ValueError(f"Expected tensor with ndim >= 2, got shape {tuple(x.shape)}")
        return x.reshape(x.shape[0], -1, x.shape[-1]).mean(dim=1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pool_features(x).detach()
        if self.use_rp:
            x = F.relu(x @ self.W_rand)
        return x

    def collect(self, x: torch.Tensor, labels: torch.Tensor) -> None:
        if labels.numel() == 0:
            return
        labels = labels.detach().long()
        if labels.min().item() < 0:
            raise ValueError("RPHead labels must be non-negative.")
        if labels.max().item() >= self.max_classes:
            raise ValueError(
                f"Observed label {labels.max().item()} exceeds rp_max_classes={self.max_classes}"
            )

        encoded = self._encode(x)
        onehot = F.one_hot(labels, num_classes=self.max_classes).to(encoded.dtype)
        self.Q = self.Q + encoded.T @ onehot
        self.G = self.G + encoded.T @ encoded
        next_classes = int(labels.max().item()) + 1
        if next_classes > int(self.num_classes_seen.item()):
            self.num_classes_seen.fill_(next_classes)
        self.needs_update.fill_(True)

    @torch.no_grad()
    def update(self) -> None:
        num_classes = int(self.num_classes_seen.item())
        if not bool(self.needs_update.item()) or num_classes <= 0:
            return

        device = self.G.device
        eye = torch.eye(self.proj_dim, device=device, dtype=self.G.dtype)
        solved = torch.linalg.solve(self.G + self.ridge * eye, self.Q[:, :num_classes]).T
        self.weight.zero_()
        self.weight[:num_classes].copy_(solved)
        self.needs_update.fill_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(x)
        return encoded @ self.weight.T


class WSRHead(nn.Module):
    """Online whitened subspace router for adapter routing."""

    def __init__(
        self,
        feature_dim: int,
        subspace_k: int = 32,
        eps: float = 1e-6,
        gamma: float = 1.0,
        max_classes: int = 64,
    ) -> None:
        super().__init__()
        if feature_dim is None or feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if subspace_k <= 0:
            raise ValueError(f"subspace_k must be positive, got {subspace_k}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if max_classes <= 0:
            raise ValueError(f"max_classes must be positive, got {max_classes}")

        self.feature_dim = feature_dim
        self.subspace_k = min(int(subspace_k), int(feature_dim))
        self.subspace_dim = self.subspace_k + 1
        self.eps = eps
        self.gamma = gamma
        self.max_classes = max_classes

        self.register_buffer("mu", torch.zeros(max_classes, feature_dim))
        self.register_buffer("var", torch.zeros(max_classes, feature_dim))
        self.register_buffer("basis", torch.zeros(max_classes, feature_dim, self.subspace_dim))
        self.register_buffer("basis_dim", torch.zeros(max_classes, dtype=torch.int64))
        self.register_buffer("is_valid", torch.zeros(max_classes, dtype=torch.bool))
        self.register_buffer("num_classes_seen", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("needs_update", torch.tensor(False, dtype=torch.bool))

        self.register_buffer("_accum_label", torch.tensor(-1, dtype=torch.int64), persistent=False)
        self.register_buffer("_accum_count", torch.tensor(0, dtype=torch.int64), persistent=False)
        self.register_buffer("_accum_sum", torch.zeros(feature_dim, dtype=torch.float64), persistent=False)
        self.register_buffer(
            "_accum_sum_outer",
            torch.zeros(feature_dim, feature_dim, dtype=torch.float64),
            persistent=False,
        )

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x
        if x.ndim < 2:
            raise ValueError(f"Expected tensor with ndim >= 2, got shape {tuple(x.shape)}")
        return x.reshape(x.shape[0], -1, x.shape[-1]).mean(dim=1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._pool_features(x).detach()

    def _reset_runtime_accumulators(self) -> None:
        self._accum_label.fill_(-1)
        self._accum_count.zero_()
        self._accum_sum.zero_()
        self._accum_sum_outer.zero_()

    def _ensure_runtime_accumulators_on_cpu(self) -> None:
        if self._accum_label.device.type != "cpu":
            self._accum_label.data = self._accum_label.data.cpu()
        if self._accum_count.device.type != "cpu":
            self._accum_count.data = self._accum_count.data.cpu()
        if self._accum_sum.device.type != "cpu":
            self._accum_sum.data = self._accum_sum.data.cpu()
        if self._accum_sum_outer.device.type != "cpu":
            self._accum_sum_outer.data = self._accum_sum_outer.data.cpu()

    def collect(self, x: torch.Tensor, labels: torch.Tensor) -> None:
        if labels.numel() == 0:
            return
        # Runtime accumulators are intentionally kept on CPU so WSR statistics
        # do not consume persistent GPU memory across long CL runs.
        self._ensure_runtime_accumulators_on_cpu()
        labels = labels.detach().long()
        if labels.min().item() < 0:
            raise ValueError("WSRHead labels must be non-negative.")
        if labels.max().item() >= self.max_classes:
            raise ValueError(
                f"Observed label {labels.max().item()} exceeds ws_max_classes={self.max_classes}"
            )

        unique_labels = torch.unique(labels)
        if unique_labels.numel() != 1:
            raise RuntimeError(
                f"WSRHead currently expects one task label per collect call, got labels {unique_labels.tolist()}."
            )

        label = int(unique_labels.item())
        current_label = int(self._accum_label.item())
        if current_label >= 0 and current_label != label and int(self._accum_count.item()) > 0:
            raise RuntimeError(
                f"WSRHead cannot mix task statistics in one process. "
                f"Current accum label={current_label}, incoming label={label}."
            )

        encoded = self._encode(x).to(device=torch.device("cpu"), dtype=torch.float64)
        if current_label != label:
            self._reset_runtime_accumulators()
            self._accum_label.fill_(label)

        self._accum_sum.add_(encoded.sum(dim=0))
        self._accum_sum_outer.add_(encoded.transpose(0, 1) @ encoded)
        self._accum_count.add_(encoded.shape[0])
        next_classes = label + 1
        if next_classes > int(self.num_classes_seen.item()):
            self.num_classes_seen.fill_(next_classes)
        self.needs_update.fill_(True)

    @torch.no_grad()
    def update(self) -> None:
        label = int(self._accum_label.item())
        count = int(self._accum_count.item())
        if not bool(self.needs_update.item()) or label < 0 or count <= 0:
            return

        mu = self._accum_sum / float(count)
        if count > 1:
            cov = (self._accum_sum_outer - float(count) * torch.outer(mu, mu)) / float(count - 1)
        else:
            cov = torch.zeros_like(self._accum_sum_outer)

        cov = 0.5 * (cov + cov.transpose(0, 1))
        var = torch.diagonal(cov).clamp_min(0.0)
        whitening = torch.rsqrt(var + self.eps)
        whitened_cov = cov * torch.outer(whitening, whitening)
        whitened_cov = 0.5 * (whitened_cov + whitened_cov.transpose(0, 1))

        if count > 1 and self.subspace_k > 0:
            eigvals, eigvecs = torch.linalg.eigh(whitened_cov)
            top_k = min(self.subspace_k, eigvecs.shape[1])
            U = eigvecs[:, -top_k:]
        else:
            U = torch.zeros(self.feature_dim, 0, dtype=torch.float64, device=whitened_cov.device)

        mean_dir = mu * whitening
        basis_parts = []
        mean_norm = torch.linalg.norm(mean_dir)
        if mean_norm > self.eps:
            basis_parts.append((mean_dir / mean_norm).unsqueeze(dim=1))
        if U.shape[1] > 0:
            basis_parts.append(U)

        if basis_parts:
            augmented = torch.cat(basis_parts, dim=1)
            Q, _ = torch.linalg.qr(augmented, mode="reduced")
        else:
            Q = torch.zeros(self.feature_dim, 0, dtype=torch.float64, device=whitened_cov.device)

        basis_dim = min(Q.shape[1], self.subspace_dim)
        self.mu[label].copy_(mu.to(dtype=self.mu.dtype, device=self.mu.device))
        self.var[label].copy_(var.to(dtype=self.var.dtype, device=self.var.device))
        self.basis[label].zero_()
        if basis_dim > 0:
            self.basis[label, :, :basis_dim].copy_(Q[:, :basis_dim].to(dtype=self.basis.dtype, device=self.basis.device))
        self.basis_dim[label].fill_(basis_dim)
        self.is_valid[label].fill_(True)
        self.needs_update.fill_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(x).to(device=self.mu.device, dtype=self.mu.dtype)
        batch_size = encoded.shape[0]
        logits = torch.full(
            (batch_size, self.max_classes),
            float("-inf"),
            device=encoded.device,
            dtype=encoded.dtype,
        )
        max_classes = int(self.num_classes_seen.item())
        if max_classes <= 0:
            return logits

        for label in range(max_classes):
            if not bool(self.is_valid[label].item()):
                continue
            whitening = torch.rsqrt(self.var[label] + self.eps)
            encoded_whitened = encoded * whitening.unsqueeze(dim=0)
            norm_sq = encoded_whitened.square().sum(dim=-1)
            basis_dim = int(self.basis_dim[label].item())
            if basis_dim > 0:
                basis = self.basis[label, :, :basis_dim]
                proj_sq = (encoded_whitened @ basis).square().sum(dim=-1)
            else:
                proj_sq = torch.zeros_like(norm_sq)
            residual_energy = 1.0 - (proj_sq / (norm_sq + self.eps))
            logits[:, label] = -self.gamma * residual_energy

        return logits


# ---- Layer wrapper: base + adapter ----
class CLARELayer(nn.Module, BaseTunerLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        peft_config: CLAREConfig,
        module_config: CLAREModuleConfig,
        adapter_name: str,
        layer_name: str,
        layer_id: int,
        base_layer_name: str,
        num_adapters: int,
        num_discriminators: int
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.peft_config = peft_config
        self.module_config = module_config
        self.adapter_name = adapter_name
        self.layer_name = layer_name
        self.layer_id = layer_id
        self.base_layer_name = base_layer_name
        self.num_adapters = num_adapters
        self.num_discriminators = num_discriminators
        self.use_lora = self.module_config.func_adapter_cfg.use_lora
        self.routing_mode = (
            "ws_router" if self.peft_config.routing_mode == "whitened_subspace" else self.peft_config.routing_mode
        )
        self.routing_group = self.module_config.routing_group
        self.routing_role = self.module_config.routing_role
        self.residual_head_ema_cfg = self.module_config.residual_head_ema_cfg
        self.residual_head_ema_enabled = self.residual_head_ema_cfg is not None
        self.residual_head_ema_ratios = (
            tuple(self.residual_head_ema_cfg.ratios) if self.residual_head_ema_enabled else tuple()
        )
        self.residual_head_ema_ensemble_method = (
            self.residual_head_ema_cfg.ensemble_method if self.residual_head_ema_enabled else "mean"
        )
        self.residual_ensemble_gate_cfg = self.module_config.residual_ensemble_gate_cfg
        self.residual_ensemble_gate_enabled = self.residual_ensemble_gate_cfg is not None
        self.residual_ensemble_gate_temperature = (
            self.residual_ensemble_gate_cfg.temperature if self.residual_ensemble_gate_enabled else 1.0
        )
        self.residual_ensemble_gate_online_logit_bias = (
            self.residual_ensemble_gate_cfg.online_logit_bias if self.residual_ensemble_gate_enabled else 0.0
        )
        self.residual_ensemble_gate_detach_residual_inputs = (
            self.residual_ensemble_gate_cfg.detach_residual_inputs if self.residual_ensemble_gate_enabled else True
        )
        self._shared_routing_state: Optional[dict[str, list[int]]] = None

        self._base_layer_device = next(self.base_layer.parameters()).device
        self._base_layer_dtype = next(self.base_layer.parameters()).dtype

        def submodule_name_match(submodule_name: str, lora_module_name_list: list[str]) -> bool:
            for registered_name in lora_module_name_list:
                if submodule_name == registered_name or submodule_name.startswith(registered_name + "."):
                    return True
            return False

        # create adapters
        if self.use_lora:
            self.lora_module_name_list = []
            new_func_adapters_list = nn.ModuleList([])

            lora_func_adapter_template = LoRAFuncAdapter(self.module_config.func_adapter_cfg)
            
            for name, module in self.base_layer.named_modules():
                if not submodule_name_match(name, self.lora_module_name_list): 
                    # only conside nn.Linear
                    if isinstance(module, nn.Linear):
                        # Replace the original base layer with lora compatiable layer
                        lora_wrapped_module = LoRALinear(module, self.module_config.func_adapter_cfg)
                        name, self.base_layer = general_set_module(self.base_layer, name, lora_wrapped_module)

                        # record name of lora wrapped module
                        self.lora_module_name_list.append(name)

                        if num_adapters > 0:
                            lora_func_adapter_template.layer_wise_lora_adapters[name.replace(".", "_")] = nn.ModuleDict({
                                "lora_a" : nn.Linear(lora_wrapped_module.in_features, lora_wrapped_module.rank, bias=False),
                                "lora_b" : nn.Linear(lora_wrapped_module.rank, lora_wrapped_module.out_features, bias=False)
                            })
                    elif isinstance(module, nn.MultiheadAttention):
                        # Replace the original base layer with lora compatiable layer
                        lora_wrapped_module = LoRAMultiheadAttention(module, self.module_config.func_adapter_cfg)
                        name, self.base_layer = general_set_module(self.base_layer, name, lora_wrapped_module)

                        # record name of lora wrapped module
                        self.lora_module_name_list.append(name)

                        if num_adapters > 0:
                            lora_func_adapter_template.layer_wise_lora_adapters[name.replace(".", "_")] = nn.ModuleDict({
                                "lora_a" : nn.Linear(lora_wrapped_module.original_layer.out_proj.in_features, lora_wrapped_module.original_layer.out_proj.rank, bias=False),
                                "lora_b" : nn.Linear(lora_wrapped_module.original_layer.out_proj.rank, lora_wrapped_module.original_layer.out_proj.out_features, bias=False)
                            })
                            lora_func_adapter_template.layer_wise_lora_parameters[name.replace(".", "_")] = nn.ParameterDict({
                                "lora_a" : nn.Linear(lora_wrapped_module.in_features, lora_wrapped_module.rank, bias=False),
                                "lora_b" : nn.Linear(lora_wrapped_module.rank, lora_wrapped_module.out_features, bias=False)
                            })


            lora_func_adapter_template.to(dtype=self._base_layer_dtype, device=self._base_layer_device)

            for _ in range(num_adapters):
                new_func_adapters_list.append(copy.deepcopy(lora_func_adapter_template))

            del lora_func_adapter_template
                    
            self.clare_func_adapters = nn.ModuleDict({self.adapter_name:new_func_adapters_list})
        else:
            new_func_adapters_list = nn.ModuleList([self._create_adapter() for _ in range(num_adapters)])
            self.clare_func_adapters: nn.ModuleDict[str, nn.ModuleList[FuncAdapter]] = \
            nn.ModuleDict({self.adapter_name:new_func_adapters_list})
        self.clare_residual_head_ema_adapters = nn.ModuleDict(
            {self.adapter_name: self._create_residual_head_ema_bank(self.clare_func_adapters[self.adapter_name])}
        )
        self.clare_residual_ensemble_gates = nn.ModuleDict(
            {self.adapter_name: self._create_residual_ensemble_gate_bank(self.clare_func_adapters[self.adapter_name])}
        )


        # create discriminators
        new_discriminators_list = nn.ModuleList([self._create_discriminator() for _ in range(num_discriminators)])
        self.clare_discriminators: nn.ModuleDict[str, nn.ModuleList[Discriminator]] = \
            nn.ModuleDict({self.adapter_name:new_discriminators_list})
        if self.routing_mode == "rp_gate" and self.routing_role != "router_follower":
            self.rp_head = RPHead(
                feature_dim=self.module_config.feature_dim,
                rp_dim=self.peft_config.rp_dim,
                ridge=self.peft_config.rp_ridge,
                max_classes=self.peft_config.rp_max_classes,
            )
        if self.routing_mode == "ws_router" and self.routing_role != "router_follower":
            self.ws_head = WSRHead(
                feature_dim=self.module_config.feature_dim,
                subspace_k=self.peft_config.ws_subspace_k,
                eps=self.peft_config.ws_eps,
                gamma=self.peft_config.ws_gamma,
                max_classes=self.peft_config.ws_max_classes,
            )

        self._info_dicts: dict = {}
        self._active_task: int = -1
        self._forwarded_adapter_id: int = -1
        self._forwarded_discriminator_id: int = -1
        self._train_discriminator: bool = False
        self._previous_forwarded_adapter_key: Optional[Tuple[str, int]] = None
        self._stack_discriminator_once_in_eval: bool = True
        self._stacked_discriminator = {}

    def _create_adapter(self):
        if self.module_config.use_trainable_copy:
            adapter = copy.deepcopy(self.base_layer)
        else:
            adapter = FuncAdapter(
                self.module_config.func_adapter_cfg, 
                self.module_config.feature_dim, 
                self.module_config.out_feature_dim
            )
        for p in adapter.parameters():
            p.requires_grad = True
        return adapter
    
    def _create_discriminator(self):
        disc_cls = get_discriminaor_class(self.module_config.discriminator_cfg.type)
        new_dis = disc_cls(self.module_config.discriminator_cfg, self.module_config.feature_dim)
        new_dis.to(device=self._base_layer_device, dtype=self._base_layer_dtype)
        return new_dis

    def set_shared_routing_state(self, shared_routing_state: dict[str, list[int]]) -> None:
        self._shared_routing_state = shared_routing_state

    def _cache_routed_adapter_ids(self, adapter_ids: list[int]) -> None:
        if self.routing_group is None or self._shared_routing_state is None:
            return
        self._shared_routing_state[self.routing_group] = list(adapter_ids)

    def _get_cached_routed_adapter_ids(self) -> list[int]:
        if self.routing_group is None:
            raise RuntimeError("router_follower requires a routing_group to reuse owner decisions.")
        if self._shared_routing_state is None or self.routing_group not in self._shared_routing_state:
            raise RuntimeError(
                f"Missing shared routing decisions for group '{self.routing_group}'. "
                "Ensure the router owner runs before the router follower."
            )
        return list(self._shared_routing_state[self.routing_group])

    def _freeze_adapter_module(self, adapter: nn.Module) -> None:
        for param in adapter.parameters():
            param.requires_grad = False
        adapter.eval()

    def _create_residual_head_ema_group(self, source_adapter: nn.Module) -> nn.ModuleList:
        ema_group = nn.ModuleList()
        if not self.residual_head_ema_enabled:
            return ema_group

        for _ in self.residual_head_ema_ratios:
            ema_adapter = copy.deepcopy(source_adapter)
            self._freeze_adapter_module(ema_adapter)
            ema_group.append(ema_adapter)
        return ema_group

    def _create_residual_head_ema_bank(self, source_adapters: nn.ModuleList) -> nn.ModuleList:
        return nn.ModuleList(
            [self._create_residual_head_ema_group(source_adapter) for source_adapter in source_adapters]
        )

    def _create_residual_ensemble_gate(self) -> Optional[nn.Module]:
        if not self.residual_ensemble_gate_enabled:
            return None
        input_dim = self.module_config.feature_dim + self.module_config.out_feature_dim * (
            1 + len(self.residual_head_ema_ratios)
        )
        gate = ResidualEnsembleGate(
            input_dim=input_dim,
            hidden_dim=self.residual_ensemble_gate_cfg.hidden_dim,
            output_dim=len(self.residual_head_ema_ratios),
        )
        gate.to(device=self._base_layer_device, dtype=self._base_layer_dtype)
        return gate

    def _create_residual_ensemble_gate_bank(self, source_adapters: nn.ModuleList) -> nn.ModuleList:
        if not self.residual_ensemble_gate_enabled:
            return nn.ModuleList()
        return nn.ModuleList([self._create_residual_ensemble_gate() for _ in source_adapters])

    def _append_adapter_with_residual_head_ema(self, new_adapter: nn.Module) -> None:
        self.clare_func_adapters[self.adapter_name].append(new_adapter)
        self.clare_residual_head_ema_adapters[self.adapter_name].append(
            self._create_residual_head_ema_group(new_adapter)
        )
        if self.residual_ensemble_gate_enabled:
            self.clare_residual_ensemble_gates[self.adapter_name].append(self._create_residual_ensemble_gate())
        self.num_adapters += 1

    def _ensemble_residual_head_outputs(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        if len(outputs) == 1:
            return outputs[0]
        if self.residual_head_ema_ensemble_method != "mean":
            raise ValueError(
                f"Unsupported residual_head_ema_ensemble_method: {self.residual_head_ema_ensemble_method}. "
                "Only 'mean' is currently implemented."
            )
        return torch.stack(outputs, dim=0).mean(dim=0)

    def _build_residual_ensemble_gate_input(
        self,
        hidden_input: torch.Tensor,
        residual_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        gate_pieces = [hidden_input]
        for residual_output in residual_outputs:
            if self.residual_ensemble_gate_detach_residual_inputs:
                residual_output = residual_output.detach()
            gate_pieces.append(residual_output)
        return torch.cat(gate_pieces, dim=-1)

    def _apply_residual_ensemble_gate(
        self,
        hidden_input: torch.Tensor,
        residual_outputs: list[torch.Tensor],
        gate_module: nn.Module,
    ) -> torch.Tensor:
        gate_input = self._build_residual_ensemble_gate_input(hidden_input, residual_outputs)
        gate_logits = gate_module(gate_input)
        online_bias = torch.full_like(gate_logits[..., :1], self.residual_ensemble_gate_online_logit_bias)
        stacked_logits = torch.cat([online_bias, gate_logits], dim=-1)
        weights = torch.softmax(stacked_logits / self.residual_ensemble_gate_temperature, dim=-1)
        residual_stack = torch.stack(residual_outputs, dim=-2)
        fused_output = torch.sum(weights.unsqueeze(dim=-1) * residual_stack, dim=-2)
        detached_weights = weights.detach()
        detached_logits = stacked_logits.detach()
        if "residual_gate_weights" in self._info_dicts:
            self._info_dicts["residual_gate_weights"].append(detached_weights)
            self._info_dicts["residual_gate_logits"].append(detached_logits)
        else:
            self._info_dicts["residual_gate_weights"] = [detached_weights]
            self._info_dicts["residual_gate_logits"] = [detached_logits]
        return fused_output

    def _ensemble_residual_outputs(
        self,
        hidden_input: torch.Tensor,
        residual_outputs: list[torch.Tensor],
        gate_module: Optional[nn.Module],
    ) -> torch.Tensor:
        if gate_module is None:
            return self._ensemble_residual_head_outputs(residual_outputs)
        return self._apply_residual_ensemble_gate(hidden_input, residual_outputs, gate_module)

    def _get_eval_adapter_outputs(
        self,
        current_input: torch.Tensor,
        adapter_input: torch.Tensor,
        adapter_batch_index: int,
        adapter_id: int,
        **kwargs,
    ) -> torch.Tensor:
        if self.use_lora:
            outputs = []
            outputs.append(
                self._forward_lora_adapter(
                    self.clare_func_adapters[self.adapter_name],
                    adapter_id,
                    ("online", adapter_id),
                    current_input,
                    **kwargs,
                )
            )
            for ema_idx, ema_collection in enumerate(
                self.clare_residual_head_ema_adapters[self.adapter_name][adapter_id]
            ):
                outputs.append(
                    self._forward_lora_adapter(
                        self.clare_residual_head_ema_adapters[self.adapter_name][adapter_id],
                        ema_idx,
                        (f"ema_{adapter_id}", ema_idx),
                        current_input,
                        **kwargs,
                    )
                )
            gate_module = None
            if self.residual_ensemble_gate_enabled:
                gate_module = self.clare_residual_ensemble_gates[self.adapter_name][adapter_id]
            return self._ensemble_residual_outputs(current_input, outputs, gate_module)

        outputs = [self.clare_func_adapters[self.adapter_name][adapter_id](current_input)]
        for ema_adapter in self.clare_residual_head_ema_adapters[self.adapter_name][adapter_id]:
            outputs.append(ema_adapter(current_input))
        gate_module = None
        if self.residual_ensemble_gate_enabled:
            gate_module = self.clare_residual_ensemble_gates[self.adapter_name][adapter_id]
        return self._ensemble_residual_outputs(current_input, outputs, gate_module)

    @torch.no_grad()
    def update_residual_head_ema(self) -> None:
        if not self.residual_head_ema_enabled or self._forwarded_adapter_id < 0:
            return

        online_adapter = self.clare_func_adapters[self.adapter_name][self._forwarded_adapter_id]
        ema_group = self.clare_residual_head_ema_adapters[self.adapter_name][self._forwarded_adapter_id]
        online_buffers = dict(online_adapter.named_buffers())
        for ema_adapter, ema_ratio in zip(ema_group, self.residual_head_ema_ratios, strict=True):
            for ema_param, online_param in zip(ema_adapter.parameters(), online_adapter.parameters(), strict=True):
                ema_param.data.mul_(ema_ratio).add_(online_param.data, alpha=1.0 - ema_ratio)

            for buffer_name, ema_buffer in ema_adapter.named_buffers():
                online_buffer = online_buffers[buffer_name]
                if torch.is_floating_point(ema_buffer):
                    ema_buffer.data.mul_(ema_ratio).add_(online_buffer.data, alpha=1.0 - ema_ratio)
                else:
                    ema_buffer.data.copy_(online_buffer.data)

    def _forward_discriminators(self, x: torch.Tensor):

        # if self._stack_discriminator_once_in_eval:
        #     new_batched_discriminator = BatchedAutoEncoderSmall(self.module_config.discriminator_cfg, self.clare_discriminators[self.adapter_name])
        #     new_batched_discriminator.to(device=self._base_layer_device, dtype=self._base_layer_dtype)
        #     self._stacked_discriminator[self.adapter_name] = new_batched_discriminator
        #     self._stack_discriminator_once_in_eval = False

        # losses, info_dicts = self._stacked_discriminator[self.adapter_name](x)

        losses = []
        info_dicts = []

        for discriminator in self.clare_discriminators[self.adapter_name]:
            loss, info_dict = discriminator(x)
            losses.append(loss)
            info_dicts.append(info_dict)

        losses = torch.stack(losses, dim=0)

        return losses, info_dicts

    def _get_adapter_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.module_config.batch_first and x.ndim == 3:
            return einops.rearrange(x, "t b d ... -> b t d ... ")
        return x

    def _get_adapter_output_shape(self, adapter_input: torch.Tensor):
        if adapter_input.ndim == 2:
            return (adapter_input.shape[0], self.module_config.out_feature_dim)
        return (adapter_input.shape[0], adapter_input.shape[1], self.module_config.out_feature_dim)

    def _to_batch_first(self, x: torch.Tensor) -> torch.Tensor:
        if not self.module_config.batch_first and x.ndim == 3:
            return einops.rearrange(x, "t b d ... -> b t d ... ")
        return x

    def _restore_output_layout(self, x: torch.Tensor) -> torch.Tensor:
        if not self.module_config.batch_first and x.ndim == 3:
            return einops.rearrange(x, "b t d ... -> t b d ... ")
        return x

    def _forward_training_residual_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = [self.clare_func_adapters[self.adapter_name][self._forwarded_adapter_id](x)]
        for ema_adapter in self.clare_residual_head_ema_adapters[self.adapter_name][self._forwarded_adapter_id]:
            outputs.append(ema_adapter(x))
        return outputs

    def _forward_training_adapter_result(self, x: torch.Tensor) -> torch.Tensor:
        if not self.residual_ensemble_gate_enabled:
            return self.clare_func_adapters[self.adapter_name][self._forwarded_adapter_id](x)
        residual_outputs = self._forward_training_residual_outputs(x)
        hidden_input = self._to_batch_first(x)
        residual_outputs = [self._to_batch_first(output) for output in residual_outputs]
        gate_module = self.clare_residual_ensemble_gates[self.adapter_name][self._forwarded_adapter_id]
        adapter_result = self._ensemble_residual_outputs(hidden_input, residual_outputs, gate_module)
        return self._restore_output_layout(adapter_result)

    def _collect_rp_stats(self, x: torch.Tensor):
        if (
            self.routing_role == "router_follower"
            or not self.peft_config.collect_rp_stats_during_train
            or self._forwarded_adapter_id < 0
        ):
            return
        adapter_input = self._get_adapter_input(x)
        labels = torch.full(
            (adapter_input.shape[0],),
            self._forwarded_adapter_id,
            device=adapter_input.device,
            dtype=torch.long,
        )
        self.rp_head.collect(adapter_input, labels)

    def _collect_wsr_stats(self, x: torch.Tensor):
        if (
            self.routing_role == "router_follower"
            or not self.peft_config.collect_wsr_stats_during_train
            or self._forwarded_adapter_id < 0
        ):
            return
        adapter_input = self._get_adapter_input(x)
        labels = torch.full(
            (adapter_input.shape[0],),
            self._forwarded_adapter_id,
            device=adapter_input.device,
            dtype=torch.long,
        )
        self.ws_head.collect(adapter_input, labels)

    def _route_with_rp_gate(self, x: torch.Tensor) -> list[int]:
        if not hasattr(self, "rp_head"):
            raise RuntimeError("RP gate routing is unavailable for router_follower modules.")
        self.rp_head.update()
        logits = self.rp_head(self._get_adapter_input(x))[:, : self.num_adapters]
        top_1_idx_list = torch.argmax(logits, dim=-1).tolist()
        self._info_dicts["rp_logits"] = logits.detach()
        self._info_dicts["top_1_idx_list"] = top_1_idx_list
        self._cache_routed_adapter_ids(top_1_idx_list)
        return top_1_idx_list

    def _route_with_wsr(self, x: torch.Tensor) -> list[int]:
        if not hasattr(self, "ws_head"):
            raise RuntimeError("WSR routing is unavailable for router_follower modules.")
        self.ws_head.update()
        logits = self.ws_head(self._get_adapter_input(x))[:, : self.num_adapters]
        top_1_idx_list = torch.argmax(logits, dim=-1).tolist()
        self._info_dicts["wsr_logits"] = logits.detach()
        self._info_dicts["top_1_idx_list"] = top_1_idx_list
        self._cache_routed_adapter_ids(top_1_idx_list)
        return top_1_idx_list

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self._info_dicts = {}
        if self.training:
            # reset the flag
            if not self._stack_discriminator_once_in_eval:
                self._stack_discriminator_once_in_eval = True
                # release previous BatchedAutoEncoderSmall
                self._stacked_discriminator.clear()
                # load discriminators into GPU again
                for discriminator in self.clare_discriminators[self.adapter_name]:
                    discriminator.to(self._base_layer_device)

            # forward specific discriminator
            if self._train_discriminator:
                _, info_dict = self.clare_discriminators[self.adapter_name][self._forwarded_discriminator_id](x)

                if self._forwarded_discriminator_id == -1:
                    discriminator_id = len(self.clare_discriminators[self.adapter_name]) - 1  
                else:
                    discriminator_id = self._forwarded_discriminator_id
                self._info_dicts[f"discriminator_{discriminator_id}"] = info_dict

                for indice, discriminator in enumerate(self.clare_discriminators[self.adapter_name]):
                    if indice != discriminator_id:
                        info_dict = {
                            "running_mean" : discriminator.running_mean,
                            "running_std" : discriminator.running_std,
                            "num_batches_tracked" : discriminator.num_batches_tracked,
                        }
                        self._info_dicts[f"discriminator_{indice}"] = info_dict

            # forward specific adapter
            if self.use_lora:
                self._activate_lora_adapter(
                    self.clare_func_adapters[self.adapter_name],
                    self._forwarded_adapter_id,
                    ("online", self._forwarded_adapter_id),
                )
                result = self.base_layer(x, **kwargs)
            else:
                adapter_result = self._forward_training_adapter_result(x)
                base_result = self.base_layer(x, **kwargs)
                result = base_result + adapter_result
            if not self._train_discriminator:
                if self.routing_mode == "rp_gate":
                    self._collect_rp_stats(x)
                elif self.routing_mode == "ws_router":
                    self._collect_wsr_stats(x)
        else:
            if self.routing_mode in {"rp_gate", "ws_router"}:
                if self.routing_role == "router_follower":
                    adapter_ids = self._get_cached_routed_adapter_ids()
                    self._info_dicts["top_1_idx_list"] = adapter_ids
                else:
                    if self.routing_mode == "rp_gate":
                        top_1_idx_list = self._route_with_rp_gate(x)
                    else:
                        top_1_idx_list = self._route_with_wsr(x)
                    adapter_ids = top_1_idx_list
            else:
                losses, info_dicts = self._forward_discriminators(x)

                for indice, info_dict in enumerate(info_dicts):
                    self._info_dicts[f"discriminator_{indice}"] = info_dict

                top_1_idx_list = torch.argmin(losses, dim=0).tolist()
                self._info_dicts["losses"] = losses.transpose(0, 1) # (n_discriminators, n_envs) -> (n_envs, n_discriminators)
                self._info_dicts["top_1_idx_list"] = top_1_idx_list
                adapter_ids = [
                    self.clare_discriminators[self.adapter_name][top_1_idx].connected_adapter_indices.item()
                    for top_1_idx in top_1_idx_list
                ]

            adapter_input = self._get_adapter_input(x)
            adapter_output_shape = self._get_adapter_output_shape(adapter_input)

            # Process each sample individually
            adapter_result = torch.zeros(adapter_output_shape, device=adapter_input.device, dtype=adapter_input.dtype)

            for idx, _forwarded_adapter_id in enumerate(adapter_ids):
                # Select single sample while preserving dims
                current_input = adapter_input[idx]
                
                # Process this sample with its best adapter
                if self.use_lora:
                    adapter_result[idx] = self._get_eval_adapter_outputs(
                        current_input=current_input,
                        adapter_input=adapter_input,
                        adapter_batch_index=idx,
                        adapter_id=_forwarded_adapter_id,
                        **kwargs,
                    )
                else:
                    adapter_result[idx] = self._get_eval_adapter_outputs(
                        current_input=current_input,
                        adapter_input=adapter_input,
                        adapter_batch_index=idx,
                        adapter_id=_forwarded_adapter_id,
                    )

            if not self.module_config.batch_first and adapter_result.ndim == 3:
                adapter_result = einops.rearrange(adapter_result, "b t d ... -> t b d ... ")
        
            if self.use_lora:
                result = adapter_result
            else:
                base_result = self.base_layer(x, **kwargs)
                result = base_result + adapter_result

        return result

    def _forward_lora_adapter(
        self,
        adapter_collection: nn.ModuleList,
        adapter_id: int,
        adapter_key: tuple[str, int],
        current_input: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        self._activate_lora_adapter(adapter_collection, adapter_id, adapter_key)
        if self.module_config.batch_first:
            current_input = current_input.unsqueeze(dim=0) # (T, D) -> (1, T, D) / (D) -> (1, D)
        else:
            current_input = current_input.unsqueeze(dim=-2) # (T, D) -> (T, 1, D) / (D) -> (1, D)
        current_output = self.base_layer(current_input, **kwargs)
        if self.module_config.batch_first:
            current_output = current_output.squeeze(dim=0) # (1, T, D) -> (T, D) / (1, D) -> (D)
        else:
            current_output = current_output.squeeze(dim=1) # (T, 1, D) -> (T, D) / (1, D) -> (D)
        return current_output

    def _activate_lora_adapter(
        self,
        adapter_collection: nn.ModuleList,
        adapter_id: int,
        adapter_key: tuple[str, int],
    ):
        # only reload adapters when switching adapters
        if self._previous_forwarded_adapter_key != adapter_key:
            # reload adapters
            for sub_module_name in self.lora_module_name_list:
                sub_module = general_get_module(self.base_layer, sub_module_name)
                sub_module.set_lora_adapter(adapter_collection[adapter_id], sub_module_name.replace(".", "_"))

            # update cache key
            self._previous_forwarded_adapter_key = adapter_key

    def add_adapter_and_discriminator(self, new_task_id:int):
        if self.use_lora:
            new_adapter = LoRAFuncAdapter(self.module_config.func_adapter_cfg)
            new_adapter.task_id = torch.tensor(new_task_id, dtype=torch.int64)
            for sub_module_name in self.lora_module_name_list:
                sub_module = general_get_module(self.base_layer, sub_module_name)
                in_features = sub_module.in_features
                out_features = sub_module.out_features
                rank = self.module_config.func_adapter_cfg.lora_rank

                if isinstance(sub_module, LoRALinear):
                    new_adapter.layer_wise_lora_adapters[sub_module_name.replace(".", "_")] = nn.ModuleDict({
                        "lora_a" : nn.Linear(in_features, rank, bias=False),
                        "lora_b" : nn.Linear(rank, out_features, bias=False)
                    })
                elif isinstance(sub_module, LoRAMultiheadAttention):
                    new_adapter.layer_wise_lora_adapters[sub_module_name.replace(".", "_")] = nn.ModuleDict({
                        "lora_a" : nn.Linear(sub_module.original_layer.out_proj.in_features, sub_module.original_layer.out_proj.rank, bias=False),
                        "lora_b" : nn.Linear(sub_module.original_layer.out_proj.rank, sub_module.original_layer.out_proj.out_features, bias=False)
                    })
                    new_adapter.layer_wise_lora_parameters[sub_module_name.replace(".", "_")] = nn.ParameterDict({
                        "lora_a" : nn.Linear(in_features, rank, bias=False),
                        "lora_b" : nn.Linear(rank, out_features, bias=False)
                    })
        else:
            new_adapter = self._create_adapter()
            new_adapter.task_id = torch.tensor(new_task_id, dtype=torch.int64)
        
        new_adapter.to(device=self._base_layer_device, dtype=self._base_layer_dtype)
        self._append_adapter_with_residual_head_ema(new_adapter)
        adapter_parameter = list(new_adapter.parameters())
        if self.residual_ensemble_gate_enabled:
            gate_module = self.clare_residual_ensemble_gates[self.adapter_name][-1]
            adapter_parameter.extend(gate_module.parameters())
            
        discriminator_parameter = self.add_discriminator(self.num_adapters - 1, new_task_id)

        return adapter_parameter, discriminator_parameter

    def add_adapter_only(self, new_task_id: int):
        if self.use_lora:
            new_adapter = LoRAFuncAdapter(self.module_config.func_adapter_cfg)
            new_adapter.task_id = torch.tensor(new_task_id, dtype=torch.int64)
            for sub_module_name in self.lora_module_name_list:
                sub_module = general_get_module(self.base_layer, sub_module_name)
                in_features = sub_module.in_features
                out_features = sub_module.out_features
                rank = self.module_config.func_adapter_cfg.lora_rank

                if isinstance(sub_module, LoRALinear):
                    new_adapter.layer_wise_lora_adapters[sub_module_name.replace(".", "_")] = nn.ModuleDict({
                        "lora_a": nn.Linear(in_features, rank, bias=False),
                        "lora_b": nn.Linear(rank, out_features, bias=False),
                    })
                elif isinstance(sub_module, LoRAMultiheadAttention):
                    new_adapter.layer_wise_lora_adapters[sub_module_name.replace(".", "_")] = nn.ModuleDict({
                        "lora_a": nn.Linear(sub_module.original_layer.out_proj.in_features, sub_module.original_layer.out_proj.rank, bias=False),
                        "lora_b": nn.Linear(sub_module.original_layer.out_proj.rank, sub_module.original_layer.out_proj.out_features, bias=False),
                    })
                    new_adapter.layer_wise_lora_parameters[sub_module_name.replace(".", "_")] = nn.ParameterDict({
                        "lora_a": nn.Linear(in_features, rank, bias=False),
                        "lora_b": nn.Linear(rank, out_features, bias=False),
                    })
        else:
            new_adapter = self._create_adapter()
            new_adapter.task_id = torch.tensor(new_task_id, dtype=torch.int64)

        new_adapter.to(device=self._base_layer_device, dtype=self._base_layer_dtype)
        self._append_adapter_with_residual_head_ema(new_adapter)
        adapter_parameter = list(new_adapter.parameters())
        if self.residual_ensemble_gate_enabled:
            gate_module = self.clare_residual_ensemble_gates[self.adapter_name][-1]
            adapter_parameter.extend(gate_module.parameters())
        return adapter_parameter

    def add_discriminator(self, connected_adapter_indices:int, new_task_id:int):
        new_discriminator = self._create_discriminator()
        new_discriminator.task_id = torch.tensor(new_task_id, dtype=torch.int64)
        new_discriminator.connected_adapter_indices = torch.tensor(connected_adapter_indices, dtype=torch.int64)
        new_discriminator.connected_adapter_task_id = self.clare_func_adapters[self.adapter_name][connected_adapter_indices].task_id
        new_discriminator.to(device=self._base_layer_device, dtype=self._base_layer_dtype)
        self.clare_discriminators[self.adapter_name].append(new_discriminator)
        self.num_discriminators += 1

        discriminator_parameter = list(new_discriminator.parameters())

        return discriminator_parameter

    def train_discriminator(self, train_discriminator:bool):
        self._train_discriminator = train_discriminator

    def track_z_score(self, require_z_score:bool):
        for discriminator in self.clare_discriminators[self.adapter_name]:
            discriminator.require_z_score = require_z_score

    def update_stats(self, require_update_stats:bool):
        if self.num_discriminators == 0 or self._forwarded_discriminator_id < 0:
            return
        self.clare_discriminators[self.adapter_name][self._forwarded_discriminator_id].require_update_stats = require_update_stats
        # for discriminator in self.clare_discriminators[self.adapter_name]:
        #     discriminator.require_update_stats = require_update_stats

    def get_adapter_id_by_discriminator_id(self, discriminator_id):
        return self.clare_discriminators[self.adapter_name][discriminator_id].connected_adapter_indices.item()

    def update_router(self):
        if hasattr(self, "rp_head"):
            self.rp_head.update()
        if hasattr(self, "ws_head"):
            self.ws_head.update()

    def update_rp_head(self):
        self.update_router()
    
    @property
    def info_dicts(self):
        return self._info_dicts
    
    def __getattr__(self, name):
        # First, try normal behavior (important!)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        # Then search inside base_layer
        if hasattr(self.base_layer, name):
            return getattr(self.base_layer, name)

        # Attribute not found
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

