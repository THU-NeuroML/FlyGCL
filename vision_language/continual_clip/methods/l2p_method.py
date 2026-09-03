import torch
from omegaconf import DictConfig

from ._prompt_base import _PromptMethodBase  # sys.path to external/ injected by _prompt_base
from models.zoo import L2P  # noqa: E402


class L2PMethod(_PromptMethodBase):
    """
    Learning to Prompt (L2P) integrated with CLIP ViT backbone.
    prompt_param: [pool_size, prompt_length, top_k_flag]  (e.g. [30, 20, 1])
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        prompt_param = list(cfg.get("prompt_param", [30, 20, 1]))
        self.prompt = L2P(
            self._feat_dim,
            cfg.task_num,
            prompt_param,
            key_dim=self._feat_dim,
            num_layers=self._num_visual_layers,
        ).to(device)
        if self.use_text_prompt:
            self.text_prompt = L2P(
                self._text_feat_dim,
                cfg.task_num,
                prompt_param,
                key_dim=self._text_feat_dim,
                num_layers=self._num_text_layers,
            ).to(device)


class L2PCodaStyleMethod(L2PMethod):
    """Alias class for explicit coda-style L2P naming in registry/config."""
    pass
