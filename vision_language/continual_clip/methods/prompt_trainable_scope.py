"""Trainable-scope helpers for prompt parameters.

Prompt methods freeze the CLIP backbone and train prompt tensors.  DualPrompt
prefix-tuning prompts have a natural two-channel layout: prefix K and prefix V.
The selection table ``e_k_*`` is the trainable routing-key matrix used to choose
prompts.  There is no trainable prompt-query tensor in the current architecture;
for matrix sweeps, ``q`` is accepted as an alias for this routing-key table.
"""

from typing import Iterable, Set

import torch
import torch.nn as nn


def _clear_prompt_mask(param: nn.Parameter) -> None:
    for attr in ("_prompt_trainable_mask", "_prompt_frozen_value", "_prompt_mask_hook"):
        value = getattr(param, attr, None)
        if attr == "_prompt_mask_hook" and value is not None:
            try:
                value.remove()
            except Exception:
                pass
        if hasattr(param, attr):
            delattr(param, attr)


def _set_masked_trainable(param: nn.Parameter, mask: torch.Tensor) -> None:
    _clear_prompt_mask(param)
    param.requires_grad = True
    param._prompt_trainable_mask = mask.detach()
    param._prompt_frozen_value = param.detach().clone()

    def _mask_grad(grad):
        return grad * param._prompt_trainable_mask.to(dtype=grad.dtype, device=grad.device)

    param._prompt_mask_hook = param.register_hook(_mask_grad)


def _parse_prompt_scope(scope: str) -> Set[str]:
    scope = str(scope or "all").lower().strip()
    scope = scope.replace("-", "_").replace("+", "")
    aliases = {
        "full": "all",
        "prompt_all": "all",
        "prompt_full": "all",
        "prompt_q": "q",
        "prompt_k": "k",
        "prompt_v": "v",
        "prompt_qk": "qk",
        "prompt_qv": "qv",
        "prompt_kv": "kv",
        "prompt_qkv": "qkv",
        "routing_key": "q",
        "route_key": "q",
        "routing": "q",
        "prompt_routing_key": "q",
        "prefix_k": "k",
        "prompt_prefix_k": "k",
        "prefix_v": "v",
        "prompt_prefix_v": "v",
        "prefix_kv": "kv",
        "prompt_prefix_kv": "kv",
    }
    scope = aliases.get(scope, scope)
    if scope == "all":
        return {"q", "k", "v"}
    if scope in {"q", "k", "v", "qk", "qv", "kv", "qkv"}:
        return set(scope)
    raise ValueError(
        f"Unsupported prompt_trainable_scope={scope!r}. "
        "Use all or prompt_q/k/v/qk/qv/kv/qkv. "
        "Here q means the prompt routing-key table, not a CLIP query projection."
    )


def _prefix_kv_mask(name: str, param: nn.Parameter, parts: Iterable[str]) -> torch.Tensor:
    selected = set(parts)
    mask = torch.zeros_like(param, dtype=param.dtype, device=param.device)
    if name.startswith("g_p_") and param.ndim >= 3 and int(param.shape[0]) == 2:
        if "k" in selected:
            mask[0, ...] = 1
        if "v" in selected:
            mask[1, ...] = 1
        return mask
    if name.startswith("e_p_") and param.ndim >= 4 and int(param.shape[1]) == 2:
        if "k" in selected:
            mask[:, 0, ...] = 1
        if "v" in selected:
            mask[:, 1, ...] = 1
        return mask
    return torch.ones_like(param, dtype=param.dtype, device=param.device)


def apply_prompt_trainable_scope(prompt_module: nn.Module, trainable_scope: str = "all") -> None:
    """Enable only selected prompt parameter groups.

    Supported groups:
      - ``q``: routing key tensors ``e_k_*`` (there is no trainable prompt-Q).
      - ``k``: prefix-K channel of DualPrompt ``g_p_*`` / ``e_p_*`` tensors.
      - ``v``: prefix-V channel of DualPrompt ``g_p_*`` / ``e_p_*`` tensors.

    L2P uses shared prompt tokens for K and V, so K/V-only scopes are not a
    faithful split there; use DualPrompt for K/V prompt-position sweeps.
    """
    if prompt_module is None:
        return
    parts = _parse_prompt_scope(trainable_scope)
    for name, param in prompt_module.named_parameters():
        _clear_prompt_mask(param)
        is_routing_key = name.startswith("e_k_")
        is_prefix_prompt = name.startswith("g_p_") or name.startswith("e_p_")
        if is_routing_key:
            param.requires_grad = "q" in parts
            continue
        if is_prefix_prompt:
            if {"k", "v"}.intersection(parts):
                mask = _prefix_kv_mask(name, param, parts)
                if torch.any(mask != 1):
                    if torch.any(mask != 0):
                        _set_masked_trainable(param, mask)
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = False
            continue
        param.requires_grad = bool(parts == {"q", "k", "v"})


def enforce_prompt_trainable_policy(module: nn.Module) -> None:
    """Restore frozen slices for masked prompt tensors after optimizer steps."""
    if module is None:
        return
    with torch.no_grad():
        for _, param in module.named_parameters():
            mask = getattr(param, "_prompt_trainable_mask", None)
            frozen = getattr(param, "_prompt_frozen_value", None)
            if mask is None or frozen is None:
                continue
            mask = mask.to(dtype=param.dtype, device=param.device)
            frozen = frozen.to(dtype=param.dtype, device=param.device)
            param.data.mul_(mask).add_(frozen * (1 - mask))
