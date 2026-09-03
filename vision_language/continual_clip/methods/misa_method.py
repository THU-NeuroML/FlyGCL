import logging
from pathlib import Path

import torch
from omegaconf import DictConfig

from .official_prompt_method import DualPromptOfficialMethod, L2POfficialMethod


def _safe_setdefault(cfg: DictConfig, key: str, value) -> None:
    if not hasattr(cfg, key) or getattr(cfg, key) is None:
        setattr(cfg, key, value)


def _unwrap_prompt_tensor(obj):
    if torch.is_tensor(obj):
        return obj.detach()
    if hasattr(obj, "data") and torch.is_tensor(obj.data):
        return obj.data.detach()
    if hasattr(obj, "prompts") and torch.is_tensor(obj.prompts):
        return obj.prompts.detach()
    if hasattr(obj, "state_dict"):
        state = obj.state_dict()
    elif isinstance(obj, dict):
        state = obj
    else:
        return None

    for key in ("prompts", "prompt", "e_prompt", "g_prompt", "weight"):
        value = state.get(key)
        if torch.is_tensor(value):
            return value.detach()

    tensors = [value.detach() for value in state.values() if torch.is_tensor(value)]
    if len(tensors) == 1:
        return tensors[0]
    return None


def _copy_prompt_like(target: torch.nn.Parameter, source: torch.Tensor) -> bool:
    source = source.to(device=target.device, dtype=target.dtype)
    if tuple(source.shape) == tuple(target.shape):
        target.data.copy_(source)
        return True

    copied = False
    dst = target.data
    if source.ndim == dst.ndim:
        slices = tuple(slice(0, min(int(a), int(b))) for a, b in zip(source.shape, dst.shape))
        dst[slices].copy_(source[slices])
        copied = True
    elif source.ndim == 3 and dst.ndim == 4 and dst.shape[1] == 2:
        # Convert patch-style [pool, len, d] prompts into K/V prefix prompts.
        pool = min(int(source.shape[0]), int(dst.shape[0]))
        length = min(int(source.shape[1]), int(dst.shape[2]))
        dim = min(int(source.shape[2]), int(dst.shape[3]))
        dst[:pool, 0, :length, :dim].copy_(source[:pool, :length, :dim])
        dst[:pool, 1, :length, :dim].copy_(source[:pool, :length, :dim])
        copied = True
    elif source.ndim == 2 and dst.ndim == 3:
        length = min(int(source.shape[0]), int(dst.shape[1]))
        dim = min(int(source.shape[1]), int(dst.shape[2]))
        dst[:, :length, :dim].copy_(source[:length, :dim].unsqueeze(0).expand(dst.shape[0], -1, -1))
        copied = True
    elif source.ndim == 3 and dst.ndim == 3 and source.shape[0] == 2 and dst.shape[0] != 2:
        # Convert prefix-style [2, len, d] into shared patch-style [len, d] prompts.
        length = min(int(source.shape[1]), int(dst.shape[1]))
        dim = min(int(source.shape[2]), int(dst.shape[2]))
        dst[:, :length, :dim].copy_(source[0, :length, :dim].unsqueeze(0).expand(dst.shape[0], -1, -1))
        copied = True
    return copied


class MISAMethod(DualPromptOfficialMethod):
    """MISA-style CLIP prompt method.

    This method combines:
      - two-sided CLIP prompt injection (vision/text controlled by prompt_modalities),
      - patch or attention-prefix injection per side,
      - MISA's non-parametric batch-level logit mask.

    ISA/FAM warmed prompts can be supplied with:
      misa_pretrained_e_prompt_path, misa_pretrained_g_prompt_path, or
      misa_pretrained_prompt_dir containing e_prompt.pt / g_prompt.pt.
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        _safe_setdefault(cfg, "prompt_modalities", "vision+text")
        _safe_setdefault(cfg, "prompt_inject", "attention_kv_prefix")
        _safe_setdefault(cfg, "vision_prompt_inject", getattr(cfg, "prompt_inject", "attention_kv_prefix"))
        _safe_setdefault(cfg, "text_prompt_inject", getattr(cfg, "prompt_inject", "attention_kv_prefix"))
        _safe_setdefault(cfg, "misa_logit_mask", True)
        super().__init__(cfg, device)
        self._method_name = "misa"
        self.misa_logit_mask = bool(getattr(cfg, "misa_logit_mask", True))
        self._load_misa_pretrained_prompts()
        self._aux_info.update({
            "method": "misa",
            "misa_logit_mask": int(self.misa_logit_mask),
            "misa_prompt_backbone": "dualprompt",
        })

    def _resolve_pretrained_path(self, kind: str):
        direct = getattr(self.cfg, f"misa_pretrained_{kind}_prompt_path", None)
        if direct:
            return Path(str(direct))
        root = getattr(self.cfg, "misa_pretrained_prompt_dir", None)
        if root:
            return Path(str(root)) / f"{kind}_prompt.pt"
        return None

    def _load_one_prompt_file(self, prompt_module, kind: str, path: Path) -> None:
        if prompt_module is None or path is None:
            return
        if not path.exists():
            logging.warning("[MISA] pretrained %s prompt path does not exist: %s", kind, path)
            return
        try:
            obj = torch.load(str(path), map_location=self.device)
            tensor = _unwrap_prompt_tensor(obj)
        except Exception as exc:
            logging.warning("[MISA] failed to load %s prompt from %s: %s", kind, path, exc)
            return
        if tensor is None:
            logging.warning("[MISA] no tensor-like %s prompt found in %s", kind, path)
            return

        loaded = []
        prefix = f"{kind}_p_"
        for name, param in prompt_module.named_parameters():
            if name.startswith(prefix) and _copy_prompt_like(param, tensor):
                loaded.append(name)
        if loaded:
            logging.info("[MISA] loaded %s prompt from %s into %s", kind, path, loaded)
        else:
            logging.warning(
                "[MISA] skipped %s prompt from %s; shape %s did not match any %s parameters",
                kind,
                path,
                tuple(tensor.shape),
                prompt_module.__class__.__name__,
            )

    def _load_misa_pretrained_prompts(self) -> None:
        for prompt_module in (self.prompt, self.text_prompt):
            self._load_one_prompt_file(prompt_module, "e", self._resolve_pretrained_path("e"))
            self._load_one_prompt_file(prompt_module, "g", self._resolve_pretrained_path("g"))

    def apply_batch_logit_mask(self, logits: torch.Tensor, local_labels: torch.Tensor) -> torch.Tensor:
        if not self.misa_logit_mask:
            return logits
        mask = torch.full_like(logits, float("-inf"))
        for c in torch.unique(local_labels.long()):
            mask[:, c] = 0.0
        return logits + mask


class MISAL2PMethod(L2POfficialMethod):
    """L2P-backed MISA variant for ablations."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        _safe_setdefault(cfg, "prompt_modalities", "vision+text")
        _safe_setdefault(cfg, "misa_logit_mask", True)
        super().__init__(cfg, device)
        self._method_name = "misa_l2p"
        self.misa_logit_mask = bool(getattr(cfg, "misa_logit_mask", True))

    def apply_batch_logit_mask(self, logits: torch.Tensor, local_labels: torch.Tensor) -> torch.Tensor:
        if not self.misa_logit_mask:
            return logits
        mask = torch.full_like(logits, float("-inf"))
        for c in torch.unique(local_labels.long()):
            mask[:, c] = 0.0
        return logits + mask
