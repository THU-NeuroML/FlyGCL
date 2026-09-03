import clip
import torch
from omegaconf import DictConfig

from ..utils import get_class_ids_per_task, get_class_names
from .base import CLMethod
from .clip_ft_base import (
    apply_clip_ft_trainable_scope,
    enforce_clip_ft_trainable_policy,
    load_clip_for_full_finetuning,
)


class CLIPFullFineTuneMethod(CLMethod):
    """Plain CLIP full fine-tuning baseline with selectable trainable scope."""

    def __init__(self, cfg: DictConfig, device: torch.device, jit: bool = False):
        super().__init__()
        self.cfg = cfg
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None

        self.model, self.transforms = load_clip_for_full_finetuning(cfg, device=device, jit=jit)
        self.freeze_text_encoder = bool(getattr(cfg, "freeze_text_encoder", False))
        self.clip_ft_trainable_scope = str(getattr(cfg, "clip_ft_trainable_scope", "full"))
        torch.save(self.model.state_dict(), "ori_state.pth")

        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.current_task = -1
        self.reset = bool(getattr(cfg, "reset", False))

        self._aux_info = {
            "method": "clip_ft",
            "freeze_text_encoder": self.freeze_text_encoder,
            "clip_ft_trainable_scope": self.clip_ft_trainable_scope,
        }

    def _build_tokens(self, class_names):
        return clip.tokenize([self.prompt_template.format(c) for c in class_names]).to(self.device)

    def forward(self, image, test=False, all_test=False, return_feature=False, replay=None):
        if test:
            with torch.no_grad():
                tokens = self.all_text_tokens if all_test else self.text_tokens
                if return_feature:
                    logits, _, img_feat, txt_feat = self.model(image, tokens, return_feature=True)
                else:
                    logits, _ = self.model(image, tokens)
                probs = logits.softmax(dim=-1)
        else:
            if return_feature:
                _, _, img_feat, txt_feat = self.model(image, self.text_tokens, return_feature=True)
                return img_feat, txt_feat
            if replay is not None:
                logits, _ = self.model(image, self.text_tokens)
                txt_feat = self.model.encode_text(self.text_tokens)
                txt_feat = txt_feat / txt_feat.norm(dim=1, keepdim=True)
                replay_feat = replay / replay.norm(dim=1, keepdim=True)
                replay_logits = replay_feat @ txt_feat.t() * 100
            else:
                logits, _ = self.model(image, self.text_tokens)
            probs = logits

        self._aux_info = {
            "method": "clip_ft",
            "freeze_text_encoder": self.freeze_text_encoder,
            "clip_ft_trainable_scope": self.clip_ft_trainable_scope,
        }

        if return_feature:
            txt_feat = self.model.encode_text(self.all_text_tokens)
            return probs, img_feat, txt_feat
        if replay is not None:
            return probs, replay_logits
        return probs

    def adaptation(self, task_id: int, reset: bool = False) -> None:
        self.current_task += 1

        if reset and self.current_task > 0:
            ori_state = torch.load("ori_state.pth", map_location=self.device)
            self.model.load_state_dict(ori_state)
            apply_clip_ft_trainable_scope(
                self.model,
                self.clip_ft_trainable_scope,
                self.freeze_text_encoder,
            )

        self.current_task_class_names = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.current_class_names += self.current_task_class_names
        self.text_tokens = self._build_tokens(self.current_class_names)
        self.current_task_text_tokens = self._build_tokens(self.current_task_class_names)
        if self.current_task == 0:
            class_names = []
            for i in range(self.cfg.task_num):
                class_names += get_class_names(self.classes_names, self.class_ids_per_task[i])
            self.all_class_names = class_names
            self.all_text_tokens = self._build_tokens(self.all_class_names)

    def on_optimizer_step(self) -> None:
        enforce_clip_ft_trainable_policy(self.model)

    def auxiliary_loss(self):
        return None

    def auxiliary_info(self):
        return dict(self._aux_info)
