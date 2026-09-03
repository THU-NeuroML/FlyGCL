import types

import clip
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig

from peft.lora import lora_clip
from ..utils import get_class_ids_per_task, get_class_names
from .base import CLMethod


def _forward_clip(self, image, text, return_feature=False):
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


class LoRAMethod(CLMethod):
    def __init__(self, cfg: DictConfig, device: torch.device, jit: bool = False):
        super().__init__()
        self.cfg = cfg
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None

        self.model, self.transforms = lora_clip.load(
            cfg.model_name, device=device, jit=jit, r=cfg.lora_rank, lora_mode=cfg.lora_mode
        )
        self.model.forward = types.MethodType(_forward_clip, self.model)

        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.current_task = -1
        self.only_reset_B = cfg.only_reset_B
        self.freeze_A = cfg.freeze_A

    def cur_text_features(self):
        f = self.model.encode_text(self.text_tokens)
        return f / f.norm(dim=1, keepdim=True)

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

        if return_feature:
            txt_feat = self.model.encode_text(self.all_text_tokens)
            return probs, img_feat, txt_feat
        if replay is not None:
            return probs, replay_logits
        return probs

    def adaptation(self, task_id: int, reset: bool = False) -> None:
        self.current_task += 1
        if reset and self.current_task > 0:
            ori_state = torch.load('ori_state.pth')
            if self.only_reset_B:
                now_state = self.model.state_dict()
                lora_params = {k: v for k, v in ori_state.items() if 'lora_B' in k}
                now_state.update(lora_params)
            else:
                now_state = ori_state
            self.model.load_state_dict(now_state)
        if self.freeze_A and self.current_task > 0:
            for name, param in self.model.named_parameters():
                if 'lora_A' in name:
                    param.requires_grad = False

        self.current_task_class_names = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.current_class_names += self.current_task_class_names
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        self.current_task_text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_task_class_names]
        ).to(self.device)
        if self.current_task == 0:
            class_names = []
            for i in range(self.cfg.task_num):
                class_names += get_class_names(self.classes_names, self.class_ids_per_task[i])
            self.all_class_names = class_names
            self.all_text_tokens = clip.tokenize(
                [self.prompt_template.format(c) for c in self.all_class_names]
            ).to(self.device)
