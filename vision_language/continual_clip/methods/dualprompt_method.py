import torch
from omegaconf import DictConfig

from ._prompt_base import _PromptMethodBase  # sys.path to external/ injected by _prompt_base
from models.zoo import DualPrompt  # noqa: E402


class DualPromptMethod(_PromptMethodBase):
    """
    DualPrompt integrated with CLIP ViT backbone.
    prompt_param: [e_pool_size, e_p_length, g_p_length]  (e.g. [10, 20, 6])
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        prompt_param = list(cfg.get("prompt_param", [10, 20, 6]))
        self.prompt = DualPrompt(
            self._feat_dim,
            cfg.task_num,
            prompt_param,
            key_dim=self._feat_dim,
            num_layers=self._num_visual_layers,
        ).to(device)
        if self.use_text_prompt:
            self.text_prompt = DualPrompt(
                self._text_feat_dim,
                cfg.task_num,
                prompt_param,
                key_dim=self._text_feat_dim,
                num_layers=self._num_text_layers,
            ).to(device)
