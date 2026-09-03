"""
CLIPViTAdapter: wraps CLIP's VisionTransformer to accept prompt tensors
from L2P / DualPrompt / CODA-Prompt / MISA-style methods.

Supported prompt injection strategies:
  - attention/KV prefix: prepend [Pk, Pv] to the attention key/value sequence.
  - patch prompt: insert prompt tokens into the visual token sequence.

CLIP blocks use .attn (nn.MultiheadAttention) and .ln_1/.ln_2/.mlp directly.
"""

import torch
import torch.nn as nn


def _prompt_inject_is_input_patch(prompt_inject: str) -> bool:
    return str(prompt_inject).lower() in {
        "patch",
        "patch_prompt",
        "input_patch",
        "input_patch_prompt",
        "soft_prompt",
    }


def _prompt_inject_is_block_patch(prompt_inject: str) -> bool:
    return str(prompt_inject).lower() in {
        "block_patch",
        "block_patch_prompt",
        "layer_patch",
        "layer_patch_prompt",
        "token",
        "token_prompt",
    }


def _prompt_tokens_from_prompt(prompt):
    if prompt is None:
        return None
    if isinstance(prompt, (list, tuple)):
        if not prompt:
            return None
        if len(prompt) >= 2 and prompt[1] is not None and prompt[0] is not prompt[1]:
            return torch.cat([prompt[0], prompt[1]], dim=1)
        return prompt[0]
    return prompt


def _block_forward_with_prompt(blk, x: torch.Tensor, prompt=None):
    """
    Replaces blk(x) when prompt tokens are present.
    x: (L, B, d) in CLIP's internal format.
    prompt: [Pk, Pv] each (B, n_p, d), or None.
    """
    attn_mask = blk.attn_mask
    if attn_mask is not None:
        attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)

    x_ln = blk.ln_1(x)  # (L, B, d)

    if prompt is None:
        attn_out = blk.attn(x_ln, x_ln, x_ln,
                            need_weights=False, attn_mask=attn_mask)[0]
    else:
        Pk, Pv = prompt  # each (B, n_p, d)
        # CLIP internal format is (L, B, d); permute prompt tokens accordingly
        Pk = Pk.permute(1, 0, 2)  # (n_p, B, d)
        Pv = Pv.permute(1, 0, 2)  # (n_p, B, d)
        key = torch.cat([Pk, x_ln], dim=0)   # (n_p+L, B, d)
        val = torch.cat([Pv, x_ln], dim=0)   # (n_p+L, B, d)

        if attn_mask is not None:
            n_p = Pk.shape[0]
            pad = attn_mask.new_zeros(attn_mask.shape[0], n_p)
            attn_mask = torch.cat([pad, attn_mask], dim=1)

        attn_out = blk.attn(x_ln, key, val,
                            need_weights=False, attn_mask=attn_mask)[0]

    x = x + attn_out
    x = x + blk.mlp(blk.ln_2(x))
    return x


def _block_forward_with_patch_prompt(blk, x: torch.Tensor, prompt=None):
    """
    Run one CLIP vision block with temporary patch prompt tokens.

    x is CLIP internal format (L, B, d). Prompt tokens are inserted after CLS,
    participate as normal self-attention tokens, then removed after the block so
    the external visual sequence length stays unchanged.
    """
    patch_tokens = _prompt_tokens_from_prompt(prompt)
    if patch_tokens is None:
        return _block_forward_with_prompt(blk, x, None)

    patch_tokens = patch_tokens.permute(1, 0, 2)  # (n_p, B, d)
    prompt_len = patch_tokens.shape[0]
    x_prompted = torch.cat([x[:1], patch_tokens, x[1:]], dim=0)
    x_prompted = _block_forward_with_prompt(blk, x_prompted, None)
    return torch.cat([x_prompted[:1], x_prompted[1 + prompt_len:]], dim=0)


def _first_prompt_layer(prompt) -> int:
    layers = getattr(prompt, "e_layers", None)
    if layers:
        return int(layers[0])
    return 0


class CLIPViTAdapter(nn.Module):
    """
    Wraps a CLIP VisionTransformer so that its forward() accepts the same
    (x, prompt, q, train, task_id) signature as CODA-Prompt's ViTZoo.

    Returns (tokens, prompt_loss) where tokens has shape (B, L, d).
    """

    def __init__(self, clip_visual: nn.Module, prompt_inject: str = "attention_kv_prefix"):
        super().__init__()
        self.vit = clip_visual
        self._blocks = list(self.vit.transformer.resblocks)
        self.prompt_inject = str(prompt_inject).lower()

    def forward(self, x: torch.Tensor, prompt=None, q=None,
                train: bool = False, task_id=None, lora_bank=None, expert_id=None,
                adapter_bank=None):
        vit = self.vit
        # patch embedding + positional embedding
        x = vit.conv1(x)                                     # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)           # (B, width, grid^2)
        x = x.permute(0, 2, 1)                               # (B, grid^2, width)
        cls = vit.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)                       # (B, grid^2+1, width)
        x = x + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x)

        x = x.permute(1, 0, 2)   # (L, B, d)  — CLIP internal format

        prompt_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
        input_prompt_len = 0
        if prompt is not None and _prompt_inject_is_input_patch(self.prompt_inject):
            x_block = x.permute(1, 0, 2)
            prompt_layer = _first_prompt_layer(prompt)
            if train:
                p_list, loss, x_block = prompt.forward(
                    q, prompt_layer, x_block, train=True, task_id=task_id
                )
                prompt_loss = prompt_loss + loss
            else:
                p_list, _, x_block = prompt.forward(
                    q, prompt_layer, x_block, train=False, task_id=task_id
                )
            x = x_block.permute(1, 0, 2)
            patch_tokens = _prompt_tokens_from_prompt(p_list)
            if patch_tokens is not None:
                patch_tokens = patch_tokens.permute(1, 0, 2)
                input_prompt_len = patch_tokens.shape[0]
                x = torch.cat([x[:1], patch_tokens, x[1:]], dim=0)

        for i, blk in enumerate(self._blocks):
            p_list = None
            if prompt is not None and input_prompt_len <= 0:
                # Keep the same prompt.forward contract as CODA-Prompt's ViT.
                x_block = x.permute(1, 0, 2)  # (B, L, d)
                if train:
                    p_list, loss, x_block = prompt.forward(
                        q, i, x_block, train=True, task_id=task_id
                    )
                    prompt_loss = prompt_loss + loss
                else:
                    p_list, _, x_block = prompt.forward(
                        q, i, x_block, train=False, task_id=task_id
                    )
                x = x_block.permute(1, 0, 2)

            if lora_bank is not None:
                x = lora_bank.forward_block(blk, x, i, int(expert_id), p_list)
            else:
                if _prompt_inject_is_block_patch(self.prompt_inject):
                    x = _block_forward_with_patch_prompt(blk, x, p_list)
                else:
                    x = _block_forward_with_prompt(blk, x, p_list)
                if adapter_bank is not None:
                    x = adapter_bank.forward_block(x, i, int(expert_id))

        if input_prompt_len > 0:
            x = torch.cat([x[:1], x[1 + input_prompt_len:]], dim=0)

        x = x.permute(1, 0, 2)   # (B, L, d)
        x = vit.ln_post(x)
        return x, prompt_loss
