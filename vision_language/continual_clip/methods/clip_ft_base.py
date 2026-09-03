"""Shared CLIP full fine-tuning helpers for continual learning baselines (EWC, LwF, ...)."""

import types
from typing import Tuple

import clip
import torch
import torch.nn as nn
from omegaconf import DictConfig


def forward_clip(self, image, text, return_feature=False):
    image_features = self.encode_image(image)
    text_features = self.encode_text(text)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    logit_scale = self.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    if return_feature:
        return logits_per_image, logits_per_text, image_features, text_features
    return logits_per_image, logits_per_text


def is_text_encoder_param(name: str) -> bool:
    """CLIP text-side params live outside the ``visual`` submodule."""
    return not name.startswith("visual.")


def _clear_trainable_mask(param: nn.Parameter) -> None:
    for attr in ("_clip_ft_trainable_mask", "_clip_ft_frozen_value", "_clip_ft_mask_hook"):
        value = getattr(param, attr, None)
        if attr == "_clip_ft_mask_hook" and value is not None:
            try:
                value.remove()
            except Exception:
                pass
        if hasattr(param, attr):
            delattr(param, attr)


def _make_qkv_mask(param: nn.Parameter, trainable_parts) -> torch.Tensor:
    mask = torch.zeros_like(param, dtype=param.dtype, device=param.device)
    rows = int(param.shape[0])
    if rows % 3 != 0:
        return torch.ones_like(param, dtype=param.dtype, device=param.device)
    third = rows // 3
    spans = {
        "q": (0, third),
        "k": (third, 2 * third),
        "v": (2 * third, rows),
    }
    selected = set(str(part).lower() for part in trainable_parts)
    if param.ndim == 1:
        for part in selected:
            if part in spans:
                start, end = spans[part]
                mask[start:end] = 1
    else:
        for part in selected:
            if part in spans:
                start, end = spans[part]
                mask[start:end, ...] = 1
    return mask


def _set_masked_trainable(param: nn.Parameter, mask: torch.Tensor) -> None:
    _clear_trainable_mask(param)
    param.requires_grad = True
    param._clip_ft_trainable_mask = mask.detach()
    param._clip_ft_frozen_value = param.detach().clone()

    def _mask_grad(grad):
        return grad * param._clip_ft_trainable_mask.to(dtype=grad.dtype, device=grad.device)

    param._clip_ft_mask_hook = param.register_hook(_mask_grad)


def enforce_clip_ft_trainable_policy(model: nn.Module) -> None:
    """Restore frozen slices for masked parameters after optimizer updates."""
    with torch.no_grad():
        for _, param in model.named_parameters():
            mask = getattr(param, "_clip_ft_trainable_mask", None)
            frozen = getattr(param, "_clip_ft_frozen_value", None)
            if mask is None or frozen is None:
                continue
            mask = mask.to(dtype=param.dtype, device=param.device)
            frozen = frozen.to(dtype=param.dtype, device=param.device)
            param.data.mul_(mask).add_(frozen * (1 - mask))


def apply_clip_ft_trainable_policy(model: nn.Module, freeze_text_encoder: bool) -> None:
    """Enable full fine-tuning; optionally freeze the text encoder."""
    apply_clip_ft_trainable_scope(model, "full", freeze_text_encoder)


def apply_clip_ft_trainable_scope(
    model: nn.Module,
    trainable_scope: str = "full",
    freeze_text_encoder: bool = False,
) -> None:
    """Select trainable CLIP parameters for full-weight fine-tuning baselines.

    ``attention_*`` scopes keep real CLIP weights but only update selected
    Q/K/V rows inside PyTorch MultiheadAttention ``in_proj_*`` tensors. Frozen
    rows are gradient-masked and restored after optimizer steps by
    ``enforce_clip_ft_trainable_policy``.
    """
    scope = str(trainable_scope or "full").lower()
    if scope in {"all", "full_finetune", "full-ft"}:
        scope = "full"
    scope_aliases = {
        "q": "attention_q",
        "k": "attention_k",
        "v": "attention_v",
        "qk": "attention_qk",
        "qv": "attention_qv",
        "kv": "attention_kv",
        "qkv": "attention_qkv",
        "attn_q": "attention_q",
        "attn_k": "attention_k",
        "attn_v": "attention_v",
        "attn_qk": "attention_qk",
        "attn_qv": "attention_qv",
        "attn_kv": "attention_kv",
        "attn_qkv": "attention_qkv",
        "attention-q": "attention_q",
        "attention-k": "attention_k",
        "attention-v": "attention_v",
        "attention-qk": "attention_qk",
        "attention-qv": "attention_qv",
        "attention-kv": "attention_kv",
        "attention-qkv": "attention_qkv",
        "all_attention_kv": "attention_kv",
        "all_attention_qkv": "attention_qkv",
    }
    scope = scope_aliases.get(scope, scope)
    qkv_parts_by_scope = {
        "attention_q": ("q",),
        "attention_k": ("k",),
        "attention_v": ("v",),
        "attention_qk": ("q", "k"),
        "attention_qv": ("q", "v"),
        "attention_kv": ("k", "v"),
        "attention_qkv": ("q", "k", "v"),
    }

    for name, param in model.named_parameters():
        _clear_trainable_mask(param)
        if freeze_text_encoder and is_text_encoder_param(name):
            param.requires_grad = False
        elif scope == "full":
            param.requires_grad = True
        elif scope in qkv_parts_by_scope:
            is_qkv = ".attn.in_proj_" in name
            if is_qkv:
                _set_masked_trainable(param, _make_qkv_mask(param, qkv_parts_by_scope[scope]))
            else:
                param.requires_grad = False
        else:
            raise ValueError(
                f"Unsupported clip_ft_trainable_scope={trainable_scope!r}. "
                "Use 'full' or one of attention_q/k/v/qk/qv/kv/qkv."
            )


def load_clip_for_full_finetuning(
    cfg: DictConfig, device: torch.device, jit: bool = False
) -> Tuple[nn.Module, object]:
    model, transforms = clip.load(cfg.model_name, device=device, jit=jit)
    model = model.float()
    model.forward = types.MethodType(forward_clip, model)
    freeze_text_encoder = bool(getattr(cfg, "freeze_text_encoder", False))
    trainable_scope = str(getattr(cfg, "clip_ft_trainable_scope", "full"))
    apply_clip_ft_trainable_scope(model, trainable_scope, freeze_text_encoder)
    return model, transforms
