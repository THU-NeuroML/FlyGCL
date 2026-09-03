import torch
from omegaconf import DictConfig

from ._prompt_base import _PromptMethodBase
from .official_prompt_modules import OfficialDualPromptPrompt, OfficialL2PPrompt


class L2POfficialMethod(_PromptMethodBase):
    """L2P path aligned to l2p-main prompt-pool semantics."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        visual_layers = self._resolve_prompt_layers("e_prompt_layer_idx", "vision", [0])
        self.prompt = OfficialL2PPrompt(
            emb_d=self._feat_dim,
            n_tasks=cfg.task_num,
            pool_size=int(cfg.get("prompt_pool_size", 10)),
            prompt_length=int(cfg.get("prompt_length", 10)),
            top_k=int(cfg.get("prompt_top_k", 4)),
            batchwise_prompt=bool(cfg.get("batchwise_prompt", True)),
            deep_prompt=bool(cfg.get("l2p_deep", False)),
            use_prompt_mask=bool(cfg.get("prompt_mask", False)),
            e_prompt_layer_idx=visual_layers,
            prompt_window_mode=str(cfg.get("prompt_window_mode", "hard_session")),
            prompt_eval_mode=str(cfg.get("prompt_eval_mode", "same_as_train")),
        ).to(device)
        if self.use_text_prompt:
            text_layers = self._resolve_prompt_layers("e_prompt_layer_idx", "text", [0])
            self.text_prompt = OfficialL2PPrompt(
                emb_d=self._text_feat_dim,
                n_tasks=cfg.task_num,
                pool_size=int(cfg.get("prompt_pool_size", 10)),
                prompt_length=int(cfg.get("prompt_length", 10)),
                top_k=int(cfg.get("prompt_top_k", 4)),
                batchwise_prompt=bool(cfg.get("batchwise_prompt", True)),
                deep_prompt=True,
                use_prompt_mask=bool(cfg.get("prompt_mask", False)),
                e_prompt_layer_idx=text_layers,
                key_dim=self._text_feat_dim,
                prompt_window_mode=str(cfg.get("prompt_window_mode", "hard_session")),
                prompt_eval_mode=str(cfg.get("prompt_eval_mode", "same_as_train")),
            ).to(device)
        self._apply_prompt_trainable_scope()


class DualPromptOfficialMethod(_PromptMethodBase):
    """DualPrompt path with official control fields for E/G prompts."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        visual_layers = self._resolve_prompt_layers("e_prompt_layer_idx", "vision", [2, 3, 4])
        visual_g_layers = self._resolve_prompt_layers("g_prompt_layer_idx", "vision", [0, 1])
        self.prompt = OfficialDualPromptPrompt(
            emb_d=self._feat_dim,
            n_tasks=cfg.task_num,
            e_pool_size=int(cfg.get("e_pool_size", 10)),
            e_prompt_length=int(cfg.get("e_prompt_length", 5)),
            g_prompt_length=int(cfg.get("g_prompt_length", 5)),
            top_k=int(cfg.get("prompt_top_k", 1)),
            batchwise_prompt=bool(cfg.get("batchwise_prompt", True)),
            use_prompt_mask=bool(cfg.get("prompt_mask", True)),
            use_g_prompt=bool(cfg.get("use_g_prompt", True)),
            g_prompt_layer_idx=visual_g_layers,
            use_prefix_tune_for_g_prompt=bool(cfg.get("use_prefix_tune_for_g_prompt", True)),
            use_e_prompt=bool(cfg.get("use_e_prompt", True)),
            e_prompt_layer_idx=visual_layers,
            use_prefix_tune_for_e_prompt=bool(cfg.get("use_prefix_tune_for_e_prompt", True)),
            prompt_window_mode=str(cfg.get("prompt_window_mode", "hard_session")),
            prompt_eval_mode=str(cfg.get("prompt_eval_mode", "same_as_train")),
        ).to(device)
        if self.use_text_prompt:
            text_layers = self._resolve_prompt_layers("e_prompt_layer_idx", "text", [2, 3, 4])
            text_g_layers = self._resolve_prompt_layers("g_prompt_layer_idx", "text", [0, 1])
            self.text_prompt = OfficialDualPromptPrompt(
                emb_d=self._text_feat_dim,
                n_tasks=cfg.task_num,
                e_pool_size=int(cfg.get("e_pool_size", 10)),
                e_prompt_length=int(cfg.get("e_prompt_length", 5)),
                g_prompt_length=int(cfg.get("g_prompt_length", 5)),
                top_k=int(cfg.get("prompt_top_k", 1)),
                batchwise_prompt=bool(cfg.get("batchwise_prompt", True)),
                use_prompt_mask=bool(cfg.get("prompt_mask", True)),
                use_g_prompt=bool(cfg.get("use_g_prompt", True)),
                g_prompt_layer_idx=text_g_layers,
                use_prefix_tune_for_g_prompt=bool(cfg.get("use_prefix_tune_for_g_prompt", True)),
                use_e_prompt=bool(cfg.get("use_e_prompt", True)),
                e_prompt_layer_idx=text_layers,
                use_prefix_tune_for_e_prompt=bool(cfg.get("use_prefix_tune_for_e_prompt", True)),
                key_dim=self._text_feat_dim,
                prompt_window_mode=str(cfg.get("prompt_window_mode", "hard_session")),
                prompt_eval_mode=str(cfg.get("prompt_eval_mode", "same_as_train")),
            ).to(device)
        self._apply_prompt_trainable_scope()
