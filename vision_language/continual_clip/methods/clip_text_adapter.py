import torch
import torch.nn as nn

from .clip_vit_adapter import _block_forward_with_prompt, _prompt_tokens_from_prompt


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


def _extend_text_attn_mask(mask: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Build a causal mask for [SOS, prompt tokens, original text tokens...]."""
    if mask is None or int(prompt_len) <= 0:
        return mask
    total_len = mask.shape[0] + int(prompt_len)
    extended = torch.empty(total_len, total_len, dtype=mask.dtype, device=mask.device)
    extended.fill_(float("-inf"))
    extended.triu_(1)
    return extended


def _block_forward_with_patch_prompt(blk, x: torch.Tensor, prompt=None):
    """
    Run one CLIP text block with temporary patch prompt tokens.

    x is CLIP internal format (L, B, d). Prompt tokens are inserted after SOS,
    participate as normal self-attention tokens, then removed after the block so
    the external text sequence length and EOT index stay unchanged.
    """
    if prompt is None:
        return _block_forward_with_prompt(blk, x, None)

    patch_tokens = _prompt_tokens_from_prompt(prompt)
    if patch_tokens is None:
        return _block_forward_with_prompt(blk, x, None)

    patch_tokens = patch_tokens.permute(1, 0, 2)  # (n_p, B, d)
    prompt_len = patch_tokens.shape[0]
    x_prompted = torch.cat([x[:1], patch_tokens, x[1:]], dim=0)

    old_mask = blk.attn_mask
    try:
        if old_mask is not None:
            blk.attn_mask = _extend_text_attn_mask(
                old_mask.to(dtype=x.dtype, device=x.device),
                prompt_len,
            )
        x_prompted = _block_forward_with_prompt(blk, x_prompted, None)
    finally:
        blk.attn_mask = old_mask

    return torch.cat([x_prompted[:1], x_prompted[1 + prompt_len:]], dim=0)


def _first_prompt_layer(prompt) -> int:
    layers = getattr(prompt, "e_layers", None)
    if layers:
        return int(layers[0])
    return 0


class CLIPTextTransformerAdapter(nn.Module):
    """Wrap CLIP's text transformer for K/V-prefix or patch-token prompts."""

    def __init__(self, clip_model: nn.Module, prompt_inject: str = "attention_kv_prefix"):
        super().__init__()
        self.clip_model = clip_model
        self._blocks = list(clip_model.transformer.resblocks)
        self.prompt_inject = str(prompt_inject).lower()

    def forward(self, text: torch.Tensor, prompt=None, q=None,
                train: bool = False, task_id=None):
        model = self.clip_model
        x = model.token_embedding(text)
        x = x + model.positional_embedding.to(dtype=x.dtype, device=x.device)
        x = x.permute(1, 0, 2)

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

        old_masks = []
        if input_prompt_len > 0:
            for blk in self._blocks:
                old_masks.append(blk.attn_mask)
                if blk.attn_mask is not None:
                    blk.attn_mask = _extend_text_attn_mask(
                        blk.attn_mask.to(dtype=x.dtype, device=x.device),
                        input_prompt_len,
                    )

        try:
            for i, blk in enumerate(self._blocks):
                p_list = None
                if prompt is not None and input_prompt_len <= 0:
                    x_block = x.permute(1, 0, 2)
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

                if _prompt_inject_is_block_patch(self.prompt_inject):
                    x = _block_forward_with_patch_prompt(blk, x, p_list)
                else:
                    x = _block_forward_with_prompt(blk, x, p_list)
        finally:
            if old_masks:
                for blk, old_mask in zip(self._blocks, old_masks):
                    blk.attn_mask = old_mask

        if input_prompt_len > 0:
            x = torch.cat([x[:1], x[1 + input_prompt_len:]], dim=0)

        x = x.permute(1, 0, 2)
        x = model.ln_final(x)
        return x, prompt_loss
