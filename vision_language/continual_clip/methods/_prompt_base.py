"""
Shared base for L2P / DualPrompt / CODA-Prompt methods.

All three methods share the same structure:
  - CLIPViTAdapter as the frozen backbone (query pass) + prompt-injected pass
    - CLIP image-text similarity logits over seen classes
  - An auxiliary prompt loss accumulated during the forward pass
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

import clip
from ..utils import get_class_ids_per_task, get_class_names
from .base import CLMethod
from .clip_vit_adapter import CLIPViTAdapter
from .clip_text_adapter import CLIPTextTransformerAdapter
from .prompt_utils import parse_prompt_modalities, resolve_prompt_layers
from .prompt_trainable_scope import apply_prompt_trainable_scope, enforce_prompt_trainable_policy

# Add CODA-Prompt-main to path so we can import its prompt modules
_CODA_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../external/CODA-Prompt-main"
)
if _CODA_PATH not in sys.path:
    sys.path.insert(0, _CODA_PATH)

from models.zoo import L2P, DualPrompt, CodaPrompt  # noqa: E402


class _PromptMethodBase(CLMethod):
    """
    Base class for L2P, DualPrompt, and CODA-Prompt.

    Subclasses must set self.prompt (the prompt module) in __init__.
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self._method_name = str(getattr(cfg, "method", "")).lower()
        self.device = device
        self.classes_names = None
        self.prompt_template = cfg.prompt_template

        # Load CLIP (plain, no LoRA) — backbone is frozen for prompt methods
        clip_model, self.transforms = clip.load(cfg.model_name, device=device)
        clip_model = clip_model.float()

        # Freeze all CLIP parameters
        for p in clip_model.parameters():
            p.requires_grad = False

        self.clip_model = clip_model
        default_prompt_inject = str(getattr(cfg, "prompt_inject", "attention_kv_prefix")).lower()
        self.prompt_inject = default_prompt_inject
        self.vision_prompt_inject = str(
            getattr(cfg, "vision_prompt_inject", default_prompt_inject)
        ).lower()
        self.text_prompt_inject = str(
            getattr(cfg, "text_prompt_inject", default_prompt_inject)
        ).lower()
        self.feat = CLIPViTAdapter(
            clip_model.visual,
            prompt_inject=self.vision_prompt_inject,
        )
        self.text_prompt_inject = str(
            getattr(cfg, "text_prompt_inject", self.text_prompt_inject)
        ).lower()
        self.text_feat = CLIPTextTransformerAdapter(
            clip_model,
            prompt_inject=self.text_prompt_inject,
        )

        # embed_dim of CLIP ViT-B/16 is 768 (width), projected to 512
        # We use the pre-projection width (768) as feature dim for the head
        self._feat_dim = clip_model.visual.transformer.width  # 768 for ViT-B/16
        self._text_feat_dim = clip_model.transformer.width
        self._num_visual_layers = len(self.feat._blocks)
        self._num_text_layers = len(self.text_feat._blocks)
        self.prompt_modalities = parse_prompt_modalities(cfg)
        self.use_vision_prompt = "vision" in self.prompt_modalities
        self.use_text_prompt = "text" in self.prompt_modalities

        # Growing linear head (expanded in adaptation())
        # Kept for backward compatibility with checkpoints/configs, but logits are
        # now produced by CLIP image-text similarity.
        self.head = None  # nn.Linear, created on first adaptation()
        self._text_features = None  # [num_seen_classes, clip_embed_dim], normalized
        self._text_tokens = None

        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.current_task = -1
        self._num_seen_classes = 0

        self.prompt_mask_old_logits = bool(getattr(cfg, "prompt_mask_old_logits", False))
        self.prompt_train_on_old_classes = bool(getattr(cfg, "prompt_train_on_old_classes", True))
        self.prompt_window_mode = str(getattr(cfg, "prompt_window_mode", "hard_session")).lower()
        self.prompt_eval_mode = str(getattr(cfg, "prompt_eval_mode", "same_as_train")).lower()

        # Accumulated prompt loss from last forward pass (train mode)
        self._prompt_loss = None
        self._aux_info = {}
        self._prompt_old_usage = None
        self._prompt_new_usage = None
        self._prompt_pool_size = 0
        self._historical_old_prompt_active = None
        self._cached_session_summary = None

        # Prompt module — set by subclass
        self.prompt = None
        self.text_prompt = None
        self.prompt_trainable_scope = str(getattr(cfg, "prompt_trainable_scope", "all"))

    def _apply_prompt_trainable_scope(self) -> None:
        apply_prompt_trainable_scope(self.prompt, self.prompt_trainable_scope)
        apply_prompt_trainable_scope(self.text_prompt, self.prompt_trainable_scope)

    def on_optimizer_step(self) -> None:
        enforce_prompt_trainable_policy(self.prompt)
        enforce_prompt_trainable_policy(self.text_prompt)

    def _resolve_prompt_layers(self, attr_name: str, modality: str, default_layers=None):
        num_layers = self._num_text_layers if str(modality) == "text" else self._num_visual_layers
        return resolve_prompt_layers(self.cfg, attr_name, num_layers, default_layers=default_layers)

    @torch.no_grad()
    def _refresh_text_features(self) -> None:
        if not self.current_class_names:
            self._text_features = None
            self._text_tokens = None
            return
        tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        self._text_tokens = tokens
        if self.use_text_prompt:
            self._text_features = None
            return
        text_features = self.clip_model.encode_text(tokens).float()
        self._text_features = F.normalize(text_features, dim=-1).detach()

    def _project_text(self, text_tokens: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        eot_idx = token_ids.argmax(dim=-1)
        text_features = text_tokens[torch.arange(text_tokens.shape[0], device=text_tokens.device), eot_idx]
        return text_features.float() @ self.clip_model.text_projection.float()

    def _encode_text_features(self, train: bool):
        if not self.use_text_prompt:
            if self._text_features is None:
                raise RuntimeError("Text features are unavailable. Call adaptation() before forward().")
            return self._text_features, torch.zeros(1, device=self.device)
        if self._text_tokens is None:
            raise RuntimeError("Text tokens are unavailable. Call adaptation() before forward().")
        if self.text_prompt is None:
            raise RuntimeError("Text prompt is enabled but no text_prompt module was created.")

        with torch.no_grad():
            query_tokens, _ = self.text_feat(self._text_tokens)
            eot_idx = self._text_tokens.argmax(dim=-1)
            q = query_tokens[torch.arange(query_tokens.shape[0], device=query_tokens.device), eot_idx]

        with torch.set_grad_enabled(bool(train)):
            text_tokens, prompt_loss = self.text_feat(
                self._text_tokens,
                prompt=self.text_prompt,
                q=q,
                train=bool(train),
                task_id=self.current_task,
            )
            text_features = self._project_text(text_tokens, self._text_tokens)
            return F.normalize(text_features, dim=-1), prompt_loss

    def _project_visual(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        if (
            hasattr(self.clip_model, "visual")
            and hasattr(self.clip_model.visual, "proj")
            and self.clip_model.visual.proj is not None
        ):
            return cls_tokens.float() @ self.clip_model.visual.proj.float()
        return cls_tokens.float()

    def _compute_clip_logits(self, cls_tokens: torch.Tensor, train: bool = False):
        text_features, text_prompt_loss = self._encode_text_features(train=bool(train))
        image_features = F.normalize(self._project_visual(cls_tokens), dim=-1)
        scale = self.clip_model.logit_scale.exp().clamp(max=100.0)
        return scale * image_features @ text_features.T, text_prompt_loss

    # ------------------------------------------------------------------
    # CLMethod interface
    # ------------------------------------------------------------------

    def adaptation(self, task_id: int, reset: bool = False) -> None:
        self.current_task += 1
        task_class_names = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.current_class_names += task_class_names
        n_new = len(task_class_names)
        n_old = self._num_seen_classes
        self._num_seen_classes += n_new

        # Expand linear head
        if self.head is None:
            self.head = nn.Linear(self._feat_dim, self._num_seen_classes, bias=True).to(self.device)
        else:
            old_weight = self.head.weight.data
            old_bias = self.head.bias.data
            self.head = nn.Linear(self._feat_dim, self._num_seen_classes, bias=True).to(self.device)
            self.head.weight.data[:n_old] = old_weight
            self.head.bias.data[:n_old] = old_bias

        # Notify prompt module of new task
        if self.current_task > 0 and hasattr(self.prompt, 'process_task_count'):
            self.prompt.process_task_count()
        if self.current_task > 0 and hasattr(self.text_prompt, 'process_task_count'):
            self.text_prompt.process_task_count()

        if hasattr(self.prompt, "set_routing_modes"):
            self.prompt.set_routing_modes(self.prompt_window_mode, self.prompt_eval_mode)
        if hasattr(self.text_prompt, "set_routing_modes"):
            self.text_prompt.set_routing_modes(self.prompt_window_mode, self.prompt_eval_mode)

        pool_size = int(
            getattr(self.prompt, "pool_size", getattr(self.prompt, "e_pool_size", 0))
        )
        self._prompt_pool_size = max(pool_size, 0)
        if self._prompt_pool_size > 0:
            self._prompt_old_usage = torch.zeros(self._prompt_pool_size, dtype=torch.long)
            self._prompt_new_usage = torch.zeros(self._prompt_pool_size, dtype=torch.long)
            if self._historical_old_prompt_active is None:
                self._historical_old_prompt_active = torch.zeros(self._prompt_pool_size, dtype=torch.bool)
        else:
            self._prompt_old_usage = None
            self._prompt_new_usage = None
            self._historical_old_prompt_active = None
        self._cached_session_summary = None

        self._last_valid_out_dim = n_old
        self._valid_out_dim = self._num_seen_classes
        self._refresh_text_features()

        # Precompute task-window metadata for diagnostics/logging.
        # Keep this only for official L2P/DualPrompt fields to avoid misleading CODA/other logs.
        has_official_prompt_window = self._method_name in {
            "l2p",
            "l2p_official",
            "dualprompt",
            "dualprompt_official",
            "misa",
            "misa_l2p",
        }
        prompt_top_k = int(getattr(self.cfg, "prompt_top_k", 1)) if has_official_prompt_window else None
        prompt_pool_size = (
            int(getattr(self.cfg, "prompt_pool_size", getattr(self.cfg, "e_pool_size", 0)))
            if has_official_prompt_window else None
        )
        self._aux_info = {
            "method": self._method_name,
            "task_id": int(self.current_task),
            "prompt_top_k": prompt_top_k,
            "prompt_pool_size": prompt_pool_size,
            "prompt_window_mode": self.prompt_window_mode,
            "prompt_eval_mode": self.prompt_eval_mode,
            "prompt_mask_old_logits": int(self.prompt_mask_old_logits),
            "prompt_train_on_old_classes": int(self.prompt_train_on_old_classes),
            "prompt_window_start": (int(self.current_task * prompt_top_k) if prompt_top_k is not None else None),
            "prompt_window_end": (int((self.current_task + 1) * prompt_top_k) if prompt_top_k is not None else None),
            "prompt_modalities": "+".join(sorted(self.prompt_modalities)),
            "prompt_inject": self.prompt_inject,
            "prompt_visual_inject": self.vision_prompt_inject if self.use_vision_prompt else None,
            "prompt_text_inject": self.text_prompt_inject if self.use_text_prompt else None,
            "prompt_trainable_scope": self.prompt_trainable_scope,
            "prompt_visual_layers": int(self._num_visual_layers) if self.use_vision_prompt else 0,
            "prompt_text_layers": int(self._num_text_layers) if self.use_text_prompt else 0,
        }

    def forward(self, image: torch.Tensor, test: bool = False, all_test: bool = False) -> torch.Tensor:
        self._prompt_loss = None
        if hasattr(self.prompt, "reset_stats"):
            self.prompt.reset_stats()
        if hasattr(self.text_prompt, "reset_stats"):
            self.text_prompt.reset_stats()

        # Query pass (no grad, no prompt) to get CLS token for prompt selection
        with torch.no_grad():
            tokens, _ = self.feat(image)
            q = tokens[:, 0, :]  # (B, d)

        if test:
            with torch.no_grad():
                out, _ = self.feat(image, prompt=(self.prompt if self.use_vision_prompt else None), q=q, train=False,
                                   task_id=self.current_task)
                out = out[:, 0, :]
                logits, _ = self._compute_clip_logits(out, train=False)
                return logits
        else:
            out, prompt_loss = self.feat(image, prompt=(self.prompt if self.use_vision_prompt else None), q=q, train=True,
                                         task_id=self.current_task)
            out = out[:, 0, :]
            logits, text_prompt_loss = self._compute_clip_logits(out, train=True)

            # Mask old classes (heuristic from CODA-Prompt)
            if self.prompt_mask_old_logits:
                logits[:, :self._last_valid_out_dim] = -float('inf')
            total_aux = prompt_loss + text_prompt_loss

            # Official l2p/dual semantics: optional pull-constraint term.
            has_pull_constraint = self._method_name in {
                "l2p",
                "l2p_official",
                "dualprompt",
                "dualprompt_official",
                "misa",
                "misa_l2p",
            }
            pull_enabled = bool(getattr(self.cfg, "pull_constraint", False)) if has_pull_constraint else False
            pull_coeff = float(getattr(self.cfg, "pull_constraint_coeff", 1.0)) if pull_enabled else None
            reduce_sim = None
            if hasattr(self.prompt, "get_last_reduce_sim"):
                reduce_sim = self.prompt.get_last_reduce_sim()
            if pull_enabled and reduce_sim is not None:
                total_aux = total_aux - pull_coeff * reduce_sim

            self._prompt_loss = total_aux
            self._aux_info.update({
                "prompt_loss": float(prompt_loss.detach().item()) if torch.is_tensor(prompt_loss) else float(prompt_loss),
                "text_prompt_loss": float(text_prompt_loss.detach().item()) if torch.is_tensor(text_prompt_loss) else float(text_prompt_loss),
                "aux_total": float(total_aux.detach().item()) if torch.is_tensor(total_aux) else float(total_aux),
            })
            if has_pull_constraint:
                self._aux_info.update({
                    "pull_constraint": pull_enabled,
                    "pull_constraint_coeff": pull_coeff,
                    "reduce_sim": float(reduce_sim.detach().item()) if torch.is_tensor(reduce_sim) else None,
                })
            if hasattr(self.prompt, "get_last_routing_info"):
                routing = self.prompt.get_last_routing_info()
                self._aux_info.update({
                    "prompt_routing_mode": routing.get("mode", None),
                    "visible_prompt_ids": routing.get("visible_prompt_ids", None),
                })
            if self._method_name == "coda":
                ortho_mu = float(getattr(self.prompt, "ortho_mu", 0.0))
                self._aux_info.update({
                    "ortho_mu": ortho_mu,
                    "ortho_penalty_active": bool(ortho_mu > 0.0),
                    "prompt_task_count": int(getattr(self.prompt, "task_count", 0)),
                })
            return logits

    def auxiliary_loss(self):
        return self._prompt_loss

    def auxiliary_info(self):
        return dict(self._aux_info)

    def uses_old_class_mask(self):
        return bool(self.prompt_mask_old_logits and not self.prompt_train_on_old_classes)

    @staticmethod
    def _usage_entropy(counts: torch.Tensor) -> float:
        total = float(counts.sum().item())
        if total <= 0.0:
            return 0.0
        p = counts.float() / total
        p = p[p > 0]
        return float((-(p * torch.log2(p))).sum().item())

    @staticmethod
    def _top_prompt_ids(counts: torch.Tensor, k: int = 5):
        if counts.numel() == 0:
            return []
        k = min(int(k), int(counts.numel()))
        top_vals, top_ids = torch.topk(counts, k=k, largest=True, sorted=True)
        out = []
        for pid, cnt in zip(top_ids.tolist(), top_vals.tolist()):
            if int(cnt) <= 0:
                continue
            out.append({"id": int(pid), "count": int(cnt)})
        return out

    def register_batch_label_stats(self, labels_local: torch.Tensor, old_dim: int, valid_mask: torch.Tensor = None):
        if self._prompt_pool_size <= 0:
            return
        if not hasattr(self.prompt, "get_last_selected_indices"):
            return

        idx = self.prompt.get_last_selected_indices()
        if idx is None:
            return

        idx = idx.long()
        if valid_mask is not None:
            keep = valid_mask.detach().cpu().bool()
            if keep.numel() == idx.shape[0]:
                idx = idx[keep]
        labels_local = labels_local.detach().cpu().long()
        if idx.shape[0] != labels_local.shape[0]:
            return

        old_mask = labels_local < int(old_dim)
        new_mask = ~old_mask

        if torch.any(old_mask):
            old_idx = idx[old_mask].reshape(-1)
            self._prompt_old_usage += torch.bincount(old_idx, minlength=self._prompt_pool_size)
        if torch.any(new_mask):
            new_idx = idx[new_mask].reshape(-1)
            self._prompt_new_usage += torch.bincount(new_idx, minlength=self._prompt_pool_size)

    def prompt_routing_summary(self):
        if self._prompt_pool_size <= 0:
            return {}

        summary = {
            "prompt_window_mode": self.prompt_window_mode,
            "prompt_eval_mode": self.prompt_eval_mode,
            "prompt_mask_old_logits": int(self.prompt_mask_old_logits),
            "prompt_train_on_old_classes": int(self.prompt_train_on_old_classes),
            "old_prompt_usage": self._prompt_old_usage.tolist() if self._prompt_old_usage is not None else None,
            "new_prompt_usage": self._prompt_new_usage.tolist() if self._prompt_new_usage is not None else None,
        }

        if self._prompt_old_usage is not None and self._prompt_new_usage is not None:
            combined_usage = self._prompt_old_usage + self._prompt_new_usage
            old_active = self._prompt_old_usage > 0
            new_active = self._prompt_new_usage > 0
            overlap = ((self._prompt_old_usage > 0) & (self._prompt_new_usage > 0)).sum().item()
            old_used = int(old_active.sum().item())
            new_used = int(new_active.sum().item())
            union = int((old_active | new_active).sum().item())
            overlap_ratio = float(overlap / union) if union > 0 else 0.0

            old_revisit_rate = None
            if self._historical_old_prompt_active is not None and int(old_active.sum().item()) > 0:
                revisited = int((old_active & self._historical_old_prompt_active).sum().item())
                old_revisit_rate = float(revisited / max(old_used, 1))

            never_used = torch.nonzero(combined_usage == 0).reshape(-1).tolist()
            summary.update({
                "old_prompt_saturation": float(old_used / max(self._prompt_pool_size, 1)),
                "new_prompt_saturation": float(new_used / max(self._prompt_pool_size, 1)),
                "old_new_prompt_overlap": int(overlap),
                "old_new_prompt_overlap_ratio": overlap_ratio,
                "mixed_prompt_saturation": float(overlap / max(self._prompt_pool_size, 1)),
                "old_prompt_revisit_rate": old_revisit_rate,
                "prompt_usage_count": combined_usage.tolist(),
                "prompt_usage_top": self._top_prompt_ids(combined_usage, k=5),
                "old_prompt_usage_top": self._top_prompt_ids(self._prompt_old_usage, k=5),
                "new_prompt_usage_top": self._top_prompt_ids(self._prompt_new_usage, k=5),
                "never_used_prompts": [int(x) for x in never_used],
                "prompt_usage_entropy": self._usage_entropy(combined_usage),
                "old_prompt_usage_entropy": self._usage_entropy(self._prompt_old_usage),
                "new_prompt_usage_entropy": self._usage_entropy(self._prompt_new_usage),
            })

        if hasattr(self.prompt, "get_usage_snapshot"):
            summary.update(self.prompt.get_usage_snapshot())

        self._cached_session_summary = dict(summary)

        return summary

    def commit_prompt_session_stats(self):
        if self._prompt_pool_size <= 0:
            return
        if self._prompt_old_usage is None or self._historical_old_prompt_active is None:
            return
        self._historical_old_prompt_active |= (self._prompt_old_usage > 0)
