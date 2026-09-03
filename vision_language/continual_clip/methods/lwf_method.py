import copy
import types

import clip
import torch
from torch.nn import functional as F
from omegaconf import DictConfig

from ..utils import get_class_ids_per_task, get_class_names
from .base import CLMethod
from .clip_ft_base import (
    apply_clip_ft_trainable_policy,
    forward_clip,
    load_clip_for_full_finetuning,
)


class LwFMethod(CLMethod):
    """Learning without Forgetting baseline for CLIP full fine-tuning class-incremental training."""

    def __init__(self, cfg: DictConfig, device: torch.device, jit: bool = False):
        super().__init__()
        self.cfg = cfg
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None

        self.model, self.transforms = load_clip_for_full_finetuning(cfg, device=device, jit=jit)
        self.freeze_text_encoder = bool(getattr(cfg, "freeze_text_encoder", False))
        torch.save(self.model.state_dict(), "ori_state.pth")

        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.current_task = -1
        self.reset = bool(getattr(cfg, "reset", False))

        self.distill_lambda = float(getattr(cfg, "distill_lambda", 1.0))
        self.distill_temp = float(getattr(cfg, "distill_temp", 2.0))
        self.lwf_start_task = int(getattr(cfg, "lwf_start_task", 1))
        self.lwf_primary_loss = str(getattr(cfg, "lwf_primary_loss", "ce")).lower()

        self._known_classes = 0
        self.teacher_model = None
        self.teacher_text_tokens = None

        self._kd_loss = None
        self._teacher_active = False
        self._aux_info = {
            "method": "lwf",
            "distill_lambda": self.distill_lambda,
            "distill_temp": self.distill_temp,
            "teacher_active": False,
            "kd": 0.0,
            "freeze_text_encoder": self.freeze_text_encoder,
            "lwf_primary_loss": self.lwf_primary_loss,
        }

    def _build_tokens(self, class_names):
        return clip.tokenize([self.prompt_template.format(c) for c in class_names]).to(self.device)

    def _clone_teacher_snapshot(self):
        teacher = copy.deepcopy(self.model).to(self.device)
        teacher.forward = types.MethodType(forward_clip, teacher)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        self.teacher_model = teacher
        self.teacher_text_tokens = self.text_tokens.detach().clone()

    def forward(self, image, test=False, all_test=False, return_feature=False, replay=None):
        self._kd_loss = None
        self._teacher_active = False

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

            kd_loss = logits.new_zeros(())
            teacher_ready = self.teacher_model is not None and self.teacher_text_tokens is not None
            should_distill = (
                teacher_ready
                and self.current_task >= self.lwf_start_task
                and self._known_classes > 0
            )
            if should_distill:
                with torch.no_grad():
                    teacher_logits, _ = self.teacher_model(image, self.teacher_text_tokens)
                old_classes = min(self._known_classes, teacher_logits.shape[1], logits.shape[1])
                if old_classes > 0:
                    t = self.distill_temp
                    student_old = logits[:, :old_classes]
                    teacher_old = teacher_logits[:, :old_classes]
                    kd_core = F.kl_div(
                        F.log_softmax(student_old / t, dim=1),
                        F.softmax(teacher_old / t, dim=1),
                        reduction="batchmean",
                    )
                    kd_loss = (t * t) * kd_core * self.distill_lambda
                    self._teacher_active = True

            self._kd_loss = kd_loss
            self._aux_info = {
                "method": "lwf",
                "distill_lambda": self.distill_lambda,
                "distill_temp": self.distill_temp,
                "teacher_active": self._teacher_active,
                "kd": float(kd_loss.detach().item()),
                "known_classes": int(self._known_classes),
                "freeze_text_encoder": self.freeze_text_encoder,
                "lwf_primary_loss": self.lwf_primary_loss,
            }
            probs = logits

        if return_feature:
            txt_feat = self.model.encode_text(self.all_text_tokens)
            return probs, img_feat, txt_feat
        if replay is not None:
            return probs, replay_logits
        return probs

    def adaptation(self, task_id: int, reset: bool = False) -> None:
        self.current_task += 1
        self._known_classes = len(self.current_class_names)

        if reset and self.current_task > 0:
            ori_state = torch.load("ori_state.pth", map_location=self.device)
            self.model.load_state_dict(ori_state)
            apply_clip_ft_trainable_policy(self.model, self.freeze_text_encoder)

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

    def after_task(self, train_loader=None) -> None:
        self._clone_teacher_snapshot()

    def auxiliary_loss(self):
        return self._kd_loss

    def auxiliary_info(self):
        return dict(self._aux_info)
