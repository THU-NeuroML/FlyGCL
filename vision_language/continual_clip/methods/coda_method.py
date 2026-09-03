import torch
from omegaconf import DictConfig

from ._prompt_base import _PromptMethodBase  # sys.path to external/ injected by _prompt_base
from models.zoo import CodaPrompt  # noqa: E402


class CODAMethod(_PromptMethodBase):
    """
    CODA-Prompt integrated with CLIP ViT backbone.
    prompt_param: [pool_size, prompt_length, ortho_mu]  (e.g. [100, 8, 0.0])
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        prompt_param = list(cfg.get("prompt_param", [100, 8, 0.0]))
        self.prompt = CodaPrompt(
            self._feat_dim,
            cfg.task_num,
            prompt_param,
            key_dim=self._feat_dim,
            num_layers=self._num_visual_layers,
        ).to(device)
        if self.use_text_prompt:
            self.text_prompt = CodaPrompt(
                self._text_feat_dim,
                cfg.task_num,
                prompt_param,
                key_dim=self._text_feat_dim,
                num_layers=self._num_text_layers,
            ).to(device)
