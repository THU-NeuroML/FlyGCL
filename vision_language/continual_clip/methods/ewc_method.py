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

FISHER_STORAGE_DEVICE = torch.device("cpu")


def cpu_detached_clone(tensor: torch.Tensor) -> torch.Tensor:
    """Store Fisher / parameter snapshots off the compute device."""
    return tensor.detach().to(device=FISHER_STORAGE_DEVICE, copy=True)


def accumulate_diag_fisher_from_grads(
    session_fisher: dict,
    named_trainable,
    *,
    effective_batch_size: int,
    storage_device: torch.device | None = None,
) -> dict:
    """Add batch_size * (dL/dtheta)^2 to the diagonal Fisher accumulator.

    The stored tensors live on ``storage_device`` (CPU by default). This is
    algebraically identical to GPU-resident ``zeros_like(param) += grad**2 * N``.
    Gradients are detached immediately; no graph is retained.
    """
    target = storage_device or FISHER_STORAGE_DEVICE
    batch = int(effective_batch_size)
    if batch <= 0:
        return session_fisher
    for name, param in named_trainable:
        if param.grad is None:
            continue
        grad = param.grad.detach().to(device=target, copy=True)
        sq = grad ** 2
        del grad
        if name not in session_fisher:
            session_fisher[name] = torch.zeros_like(sq, device=target)
        session_fisher[name] = session_fisher[name] + sq * batch
    return session_fisher


def normalize_session_fisher(session_fisher: dict, samples_seen: int) -> dict:
    denom = float(max(int(samples_seen), 1))
    return {
        name: value.detach().clone() / denom
        for name, value in session_fisher.items()
    }


def ewc_penalty_value(
    named_trainable,
    prev_params: dict,
    fisher: dict,
    ewc_lambda: float,
    compute_device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Online EWC quadratic penalty: lambda * sum_i F_i (theta_i - theta*_i)^2.

    Fisher and previous-parameter tensors may reside on CPU; they are moved
    lazily, parameter-by-parameter, then released.
    """
    penalty = torch.zeros((), device=compute_device)
    active_terms = 0
    for name, param in named_trainable:
        if name not in prev_params or name not in fisher:
            continue
        prev = prev_params[name].to(device=param.device, dtype=param.dtype)
        fish = fisher[name].to(device=param.device, dtype=param.dtype)
        penalty = penalty + (fish * (param - prev) ** 2).sum()
        active_terms += 1
        del prev, fish
    return penalty * float(ewc_lambda), active_terms


def merge_online_fisher(old_fisher: dict, new_fisher: dict, gamma: float) -> dict:
    if not old_fisher:
        return {name: cpu_detached_clone(value) for name, value in new_fisher.items()}
    merged = {}
    names = set(old_fisher) | set(new_fisher)
    for name in names:
        incoming = new_fisher.get(name)
        previous = old_fisher.get(name)
        if incoming is None:
            merged[name] = cpu_detached_clone(previous)
            continue
        stored = cpu_detached_clone(incoming)
        if previous is None:
            merged[name] = stored
        else:
            merged[name] = float(gamma) * previous.to(device=stored.device, dtype=stored.dtype) + stored
    return merged


def _normalize_ewc_protocol_mode(raw_mode: str) -> str:
    mode = str(raw_mode).lower()
    if mode in {"strict", "strict_session_task"}:
        return "strict_session_task"
    if mode in {"faithful", "faithful_session_task"}:
        return "faithful_session_task"
    return "strict_session_task"


class EWCMethod(CLMethod):
    """CLIP full fine-tuning EWC with selectable retention: online-compressed or taskwise constraints."""

    def __init__(self, cfg: DictConfig, device: torch.device, jit: bool = False):
        super().__init__()
        self.cfg = cfg
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None

        self.model, self.transforms = load_clip_for_full_finetuning(cfg, device=device, jit=jit)
        self.method_name = str(getattr(cfg, "method", "online_ewc")).lower()
        self.freeze_text_encoder = bool(getattr(cfg, "freeze_text_encoder", False))
        self.clip_ft_trainable_scope = str(getattr(cfg, "clip_ft_trainable_scope", "full"))
        torch.save(self.model.state_dict(), "ori_state.pth")

        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.current_task = -1
        self.reset = bool(getattr(cfg, "reset", False))

        self.ewc_lambda = float(getattr(cfg, "ewc_lambda", 10.0))
        self.ewc_gamma = float(getattr(cfg, "ewc_gamma", 1.0))
        self.fisher_n_samples = int(getattr(cfg, "fisher_n_samples", -1))
        self.fisher_batch_size = int(getattr(cfg, "fisher_batch_size", -1))
        self.ewc_start_task = int(getattr(cfg, "ewc_start_task", 1))
        self.ewc_protocol_mode = _normalize_ewc_protocol_mode(str(getattr(cfg, "ewc_protocol_mode", "strict")))
        self.ewc_retention_mode = str(getattr(cfg, "ewc_retention_mode", "online")).lower()
        if self.ewc_retention_mode not in {"online", "taskwise"}:
            self.ewc_retention_mode = "online"
        self.ewc_max_anchors = int(getattr(cfg, "ewc_max_anchors", -1))
        self._protocol_retention_overridden = False
        if self.ewc_protocol_mode == "faithful_session_task" and self.ewc_retention_mode != "taskwise":
            self.ewc_retention_mode = "taskwise"
            self._protocol_retention_overridden = True
        if self.ewc_protocol_mode == "strict_session_task" and self.ewc_retention_mode != "online":
            self.ewc_retention_mode = "online"
            self._protocol_retention_overridden = True

        self.prev_params = {}
        self.fisher = {}
        self._session_fisher = {}
        self._session_fisher_samples = 0
        self.auxiliary_independent_of_activations = True
        self.taskwise_prev_params = []
        self.taskwise_fishers = []
        self._known_classes = 0
        self._active_anchor_count = 0

        self._ewc_penalty = None
        self._fisher_active = False
        self._aux_info = {
            "method": self.method_name,
            "ewc": 0.0,
            "ewc_lambda": self.ewc_lambda,
            "fisher_active": False,
            "ewc_retention_mode": self.ewc_retention_mode,
            "ewc_protocol_mode": self.ewc_protocol_mode,
            "ewc_protocol_retention_overridden": int(bool(self._protocol_retention_overridden)),
            "retained_anchors": 0,
            "active_anchors": 0,
            "freeze_text_encoder": self.freeze_text_encoder,
            "clip_ft_trainable_scope": self.clip_ft_trainable_scope,
            "fisher_source": "online_batch_grad",
        }
        self._post_session_no_grad_passes = 0
        self._last_fisher_samples_seen = 0
        self._last_fisher_compute_time = 0.0

    def _build_tokens(self, class_names):
        return clip.tokenize([self.prompt_template.format(c) for c in class_names]).to(self.device)

    def _trainable_named_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                yield name, param

    def _snapshot_params(self):
        snap = {}
        for name, param in self._trainable_named_parameters():
            snap[name] = cpu_detached_clone(param)
        return snap

    def _constraint_pairs(self):
        if self.ewc_retention_mode == "taskwise":
            return list(zip(self.taskwise_prev_params, self.taskwise_fishers))
        if len(self.prev_params) == 0 or len(self.fisher) == 0:
            return []
        return [(self.prev_params, self.fisher)]

    def _fisher_terms_count(self):
        if self.ewc_retention_mode == "taskwise":
            return int(sum(len(f) for f in self.taskwise_fishers))
        return int(len(self.fisher))

    def accumulate_online_fisher_from_grads(self, batch_size: int) -> None:
        """Accumulate diagonal Fisher from the current training batch gradients."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            return
        if self.fisher_n_samples > 0 and self._session_fisher_samples >= self.fisher_n_samples:
            return

        effective_batch_size = batch_size
        if self.fisher_n_samples > 0:
            remaining = max(self.fisher_n_samples - self._session_fisher_samples, 0)
            effective_batch_size = min(batch_size, remaining)
            if effective_batch_size <= 0:
                return

        accumulate_diag_fisher_from_grads(
            self._session_fisher,
            self._trainable_named_parameters(),
            effective_batch_size=effective_batch_size,
            storage_device=FISHER_STORAGE_DEVICE,
        )

        self._session_fisher_samples += int(effective_batch_size)
        self._last_fisher_samples_seen = int(self._session_fisher_samples)
        self._post_session_no_grad_passes = 0
        self._last_fisher_compute_time = 0.0

    def _finalize_session_fisher(self):
        samples_seen = int(self._session_fisher_samples)
        fisher = normalize_session_fisher(self._session_fisher, samples_seen)
        self._session_fisher = {}
        self._session_fisher_samples = 0
        self._last_fisher_samples_seen = samples_seen
        self._last_fisher_compute_time = 0.0
        self._post_session_no_grad_passes = 0
        return fisher, samples_seen

    def _compute_ewc_penalty(self):
        constraints = self._constraint_pairs()
        ready = (
            self.current_task >= self.ewc_start_task
            and len(constraints) > 0
            and self._known_classes > 0
        )
        if not ready:
            self._fisher_active = False
            self._active_anchor_count = 0
            return torch.zeros((), device=self.device)

        penalty = torch.zeros((), device=self.device)
        active_terms = 0
        active_anchors = 0
        named = list(self._trainable_named_parameters())
        for prev_params, fisher in constraints:
            term, anchor_active_terms = ewc_penalty_value(
                named,
                prev_params,
                fisher,
                ewc_lambda=1.0,
                compute_device=self.device,
            )
            penalty = penalty + term
            active_terms += int(anchor_active_terms)
            if anchor_active_terms > 0:
                active_anchors += 1

        self._fisher_active = active_terms > 0
        self._active_anchor_count = int(active_anchors)
        return penalty * self.ewc_lambda

    def forward(self, image, test=False, all_test=False, return_feature=False, replay=None):
        self._ewc_penalty = None
        self._fisher_active = False

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

            ewc_penalty = self._compute_ewc_penalty()
            self._ewc_penalty = ewc_penalty
            self._aux_info = {
                "method": self.method_name,
                "ewc": float(ewc_penalty.detach().item()),
                "ewc_lambda": self.ewc_lambda,
                "fisher_active": self._fisher_active,
                "fisher_terms": self._fisher_terms_count(),
                "known_classes": int(self._known_classes),
                "ewc_retention_mode": self.ewc_retention_mode,
                "ewc_protocol_mode": self.ewc_protocol_mode,
                "ewc_protocol_retention_overridden": int(bool(self._protocol_retention_overridden)),
                "retained_anchors": int(len(constraints := self._constraint_pairs())),
                "active_anchors": int(self._active_anchor_count),
                "freeze_text_encoder": self.freeze_text_encoder,
                "clip_ft_trainable_scope": self.clip_ft_trainable_scope,
                "fisher_source": "online_batch_grad",
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
        self._session_fisher = {}
        self._session_fisher_samples = 0

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

    def after_task(self, train_loader=None) -> None:
        new_fisher, samples = self._finalize_session_fisher()
        new_anchor = self._snapshot_params()

        if self.ewc_retention_mode == "taskwise":
            self.taskwise_prev_params.append(new_anchor)
            self.taskwise_fishers.append(new_fisher)

            if self.ewc_max_anchors > 0:
                self.taskwise_prev_params = self.taskwise_prev_params[-self.ewc_max_anchors :]
                self.taskwise_fishers = self.taskwise_fishers[-self.ewc_max_anchors :]

            self.prev_params = dict(new_anchor)
            self.fisher = dict(new_fisher)
        else:
            self.fisher = merge_online_fisher(self.fisher, new_fisher, self.ewc_gamma)
            self.prev_params = new_anchor

        retained_anchors = (
            len(self.taskwise_prev_params)
            if self.ewc_retention_mode == "taskwise"
            else (1 if (len(self.prev_params) > 0 and len(self.fisher) > 0) else 0)
        )
        self._aux_info = {
            "method": self.method_name,
            "ewc": 0.0,
            "ewc_lambda": self.ewc_lambda,
            "fisher_active": self._fisher_terms_count() > 0,
            "fisher_terms": self._fisher_terms_count(),
            "fisher_samples": int(samples),
            "fisher_compute_time": float(self._last_fisher_compute_time),
            "fisher_source": "online_batch_grad",
            "known_classes": int(self._known_classes),
            "ewc_retention_mode": self.ewc_retention_mode,
            "ewc_protocol_mode": self.ewc_protocol_mode,
            "ewc_protocol_retention_overridden": int(bool(self._protocol_retention_overridden)),
            "retained_anchors": int(retained_anchors),
            "active_anchors": int(self._active_anchor_count),
            "freeze_text_encoder": self.freeze_text_encoder,
            "clip_ft_trainable_scope": self.clip_ft_trainable_scope,
        }

    def on_optimizer_step(self) -> None:
        enforce_clip_ft_trainable_policy(self.model)

    def auxiliary_loss(self):
        return self._ewc_penalty

    def auxiliary_info(self):
        return dict(self._aux_info)

    def post_session_stats(self):
        return {
            "post_session_no_grad_passes": int(self._post_session_no_grad_passes),
            "fisher_samples_seen": int(self._last_fisher_samples_seen),
            "fisher_compute_time": float(self._last_fisher_compute_time),
            "fisher_source": "online_batch_grad",
            "after_session_updates": 0,
            "ewc_retention_mode": self.ewc_retention_mode,
            "ewc_protocol_mode": self.ewc_protocol_mode,
            "ewc_protocol_retention_overridden": int(bool(self._protocol_retention_overridden)),
            "retained_anchors": int(
                len(self.taskwise_prev_params)
                if self.ewc_retention_mode == "taskwise"
                else (1 if (len(self.prev_params) > 0 and len(self.fisher) > 0) else 0)
            ),
            "freeze_text_encoder": self.freeze_text_encoder,
            "clip_ft_trainable_scope": self.clip_ft_trainable_scope,
        }
