"""FlyVGAMethod: FlyMethod (visual experts) + CLAP4CLIP-aligned text branch.

Architecture overview
---------------------
Vision side  (inherited from FlyMethod, unchanged):
  • Per-task expert bank (prompt | adapter | lora) modified by the RP-routed
    expert at test time, and by the current-task expert during training.
  • fly_mode = "prompt" | "adapter" | "lora"

Text side  (fully CLAP4CLIP-aligned):
  1. Shared VGA (Vision-Guided Attention):
       • Single TransformerDecoder(d_model=ctx_dim=512, nhead=vga_nhead, gelu)
       • Always trainable across tasks (same as CLAP4CLIP)
       • Query = [frozen_text_features, task_token_0, ..., task_token_{T-1}]
         with shape [1, K+T, ctx_dim]
       • Memory = normalised image features (detached)  [1, B, ctx_dim]
       • Attention mask: class tokens only see same-task classes + own task
         token; task token i does not see other task tokens
  2. Per-task task tokens (expandable ParameterList, CLAP4CLIP expandable_tokens):
       • Shape [1, 1, ctx_dim] each; trunc_normal_ initialized; old ones frozen.
  3. Per-task variational adapters (expandable ModuleList):
       • For task i: text_enh_i = text_frozen_i + vga_class_i + vga_task_token_i
         (the task-token VGA residual is broadcast to all classes in that task)
       • qdist = Normal(mu_i(text_enh_i), sigma_i(text_enh_i))
       • samples ~ qdist.rsample([forward_times])   [T, n_cls_i, ctx_dim]
       • text_final_i = text_enh_i + samples        [T, n_cls_i, ctx_dim]
       • logits_i = logit_scale * img_clip @ normalize(text_final_i).T  → mean over T
       • KL = kl_divergence(qdist, N(0,I)).mean(0).sum() * kl_loss_weight
         only for current task, only during training

Loss:
  CE(logits, labels)  +  kl_loss_weight * KL

Note on ctx_dim:
  ctx_dim = text_projection.shape[1] = 512 (CLIP embed dim, ViT-B/16).  VGA,
  adapters, and task tokens all operate in this post-projection 512-dim space,
  identical to CLAP4CLIP.  No additional text_projection is applied at logit
  computation time (image and text are already in the same 512-dim space).

Config knobs (all optional with sensible defaults):
  use_vga           (bool,  default True)   – enable Vision-Guided Attention
  vga_nhead         (int,   default 1)      – attention heads in VGA decoder
  vga_dropout       (float, default 0.0)    – dropout in VGA decoder
  use_task_tokens   (bool,  default True)   – CLAP4CLIP expandable_tokens
  use_variational   (bool,  default True)   – enable variational sampling
  forward_times     (int,   default 5)      – number of Monte-Carlo samples
  kl_loss_weight    (float, default 0.001)  – λ for KL auxiliary loss
  prompt_templates  (list,  default None)   – list of text templates for frozen feats
                                              (falls back to prompt_template if unset)
  prompt_template   (str,   default "a photo of a {}.") – single template fallback
  + all FlyMethod knobs (fly_mode, rp_dim, rp_ridge, ema_ratio, etc.)
"""

import math
from typing import List, Optional, Tuple

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal

from ..utils import get_class_names
from .fly_method import FlyMethod


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: simple linear (mu) or linear+softplus (sigma) adapter
# ──────────────────────────────────────────────────────────────────────────────

class _VarAdapter(nn.Module):
    """Single linear layer for mu, or linear + softplus (scaled) for sigma."""

    def __init__(self, embed_dim: int, sigma: bool = False) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        if self.sigma:
            # Strictly positive, bounded away from zero  (CLAP4CLIP style)
            out = F.softplus(out) * 0.999 + 0.001
        return out


# ──────────────────────────────────────────────────────────────────────────────
#  FlyVGAMethod
# ──────────────────────────────────────────────────────────────────────────────

class FlyVGAMethod(FlyMethod):
    """
    FlyMethod visual experts  +  CLAP4CLIP text branch (fully aligned).

    Text branch alignment to CLAP4CLIP:
      • Single shared VGA, always trainable (not per-task).
      • Per-task task tokens (expandable ParameterList).
      • CLAP4CLIP-style attention mask for task isolation.
      • Adapter heads initialized from frozen text feature covariance.
      • ctx_dim = 512 (CLIP embed dim, text_projection output).
    Vision branch:
      • Three FlyMethod modes: prompt / adapter / lora.
      • RP (random projection) routing at test time.
    """

    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        super().__init__(cfg, device)

        # ── ctx_dim: CLIP embedding dim (text_projection output dim) ───────────
        # CLAP4CLIP uses clip_model.ln_final.weight.shape[0] which equals
        # text_projection.shape[0] = text_projection.shape[1] = 512 for ViT-B/16.
        # The VGA, adapters, and task tokens all operate in this 512-dim space.
        ctx_dim = int(self.clip_model.text_projection.shape[1])   # = 512
        self._ctx_dim = ctx_dim
        self._clip_embed_dim = ctx_dim  # same value for ViT-B/16

        # ── Shared VGA (single module, always trainable) ───────────────────────
        self.use_vga = bool(getattr(cfg, "use_vga", True))
        self._vga_nhead   = int(getattr(cfg, "vga_nhead",   1))
        self._vga_dropout = float(getattr(cfg, "vga_dropout", 0.0))
        if self.use_vga:
            dec_layer = nn.TransformerDecoderLayer(
                d_model=ctx_dim,
                nhead=self._vga_nhead,
                activation="gelu",
                batch_first=True,
                dropout=self._vga_dropout,
            ).float()
            self.vga: Optional[nn.TransformerDecoder] = nn.TransformerDecoder(
                dec_layer, num_layers=1
            ).to(device)
        else:
            self.vga = None

        # ── Per-task task tokens (CLAP4CLIP expandable_tokens) ────────────────
        # Each token: [1, 1, ctx_dim]; trunc_normal_ init; old ones frozen.
        self.use_task_tokens = bool(getattr(cfg, "use_task_tokens", True))
        self.task_tokens: Optional[nn.ParameterList] = None  # init in adaptation()

        # ── Per-task variational adapters (dim = ctx_dim = 512) ───────────────
        self.mu_adapters:    nn.ModuleList = nn.ModuleList()
        self.sigma_adapters: nn.ModuleList = nn.ModuleList()
        self.use_variational = bool(getattr(cfg, "use_variational", True))
        self.forward_times   = int(getattr(cfg, "forward_times",   5))
        self.kl_loss_weight  = float(getattr(cfg, "kl_loss_weight", 0.001))

        # Number of classes per task (index-aligned with adapters / task_tokens)
        self._cls_per_task: List[int] = []

        # ── Frozen text-feature buffers (CLIP embed space, 512-dim) ────────────
        # _frozen_text_feats:            [K, 512]         mean-over-templates, renorm'd
        # _frozen_text_feats_individual: [K, n_tpl, 512]  per-template (for adapter init)
        self.register_buffer("_frozen_text_feats", torch.zeros(1, ctx_dim))
        self._frozen_text_feats_individual: Optional[torch.Tensor] = None

        # KL loss from the most recent training forward pass
        self._pending_kl_loss: Optional[torch.Tensor] = None

        self._aux_info["method"] = "fly_vga"

    # ──────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _refresh_text_features(self) -> None:
        """
        Encode ALL current_class_names in post-projection 512-dim CLIP embed space,
        matching CLAP4CLIP's prior_text_features().

        Supports multiple templates via cfg.prompt_templates (list of strings).
        Falls back to cfg.prompt_template (single string) if not set.

        Sets:
          _frozen_text_feats:            [K, 512]         mean-over-templates, renorm'd
          _frozen_text_feats_individual: [K, n_tpl, 512]  per-template features
        """
        if not self.current_class_names:
            return

        # Collect templates (CLAP4CLIP uses a list; single string also supported)
        raw_templates = getattr(self.cfg, "prompt_templates", None)
        if raw_templates is not None and len(raw_templates) > 0:
            templates = [str(t) for t in raw_templates]
        else:
            templates = [str(getattr(self.cfg, "prompt_template", "a photo of a {}."))]

        K     = len(self.current_class_names)
        n_tpl = len(templates)

        # Build [K * n_tpl] prompts in row-major order [cls0_t0, cls0_t1, ..., clsK_tN]
        all_prompts = [
            t.format(c.replace("_", " "))
            for c in self.current_class_names
            for t in templates
        ]
        tokens = clip.tokenize(all_prompts).to(self.device)            # [K*n_tpl, seq]
        feats  = self.clip_model.encode_text(tokens).float()           # [K*n_tpl, 512]
        feats  = F.normalize(feats, dim=-1)
        feats  = feats.view(K, n_tpl, -1)                              # [K, n_tpl, 512]

        # Mean over templates then renorm (CLAP4CLIP's prior_text_features)
        mean_f = F.normalize(feats.mean(dim=1), dim=-1)                # [K, 512]
        self._frozen_text_feats            = mean_f.detach()
        self._frozen_text_feats_individual = feats.detach()            # [K, n_tpl, 512]

    def _get_attention_mask(
        self,
        n_tasks: int,
        n_query: int,   # = K_total (total class text features)
    ) -> torch.Tensor:
        """
        CLAP4CLIP get_attention_mask, adapted for our task structure.

        Full query sequence: [1, K+T, ctx_dim]
          K = n_query   total class text features
          T = n_tasks   (number of task tokens, one per task)

        Masking rules  (True = ignore / mask out in self-attention):
          1. Class token j of task i masks out all class tokens of other tasks.
          2. Class token j of task i masks out all task tokens EXCEPT its own
             task token i  (i.e. it CAN attend to token i).
          3. Task token i masks out all other task tokens j != i.
          4. Task token i masks out all class tokens not belonging to task i.

        Returns: [K+T, K+T] bool tensor on self.device
        """
        T = n_tasks if self.use_task_tokens else 0
        total = n_query + T
        mask = torch.zeros(total, total, dtype=torch.bool, device=self.device)

        start_cls = 0
        for i, n_cls in enumerate(self._cls_per_task[:n_tasks]):
            end_cls = start_cls + n_cls
            cls_range = range(start_cls, end_cls)

            for cls_idx in cls_range:
                # Rule 1: mask out classes of other tasks
                mask[cls_idx, :start_cls]      = True
                mask[cls_idx, end_cls:n_query] = True
                # Rule 2: mask ALL task tokens, then unmask own task token
                if self.use_task_tokens and T > 0:
                    mask[cls_idx, n_query:n_query + T] = True
                    if i < T:
                        mask[cls_idx, n_query + i] = False   # ← unmask own token

            # Rules 3 & 4: task token i behaviour
            if self.use_task_tokens and i < T:
                ti = n_query + i
                mask[ti, n_query:n_query + i]          = True   # rule 3a
                mask[ti, n_query + i + 1:n_query + T]  = True   # rule 3b
                mask[ti, :start_cls]                   = True   # rule 4a
                mask[ti, end_cls:n_query]               = True   # rule 4b

            start_cls = end_cls

        return mask

    def _init_new_adapter_heads(self) -> None:
        """
        Initialise newly added adapter weights from frozen text feature covariance.
        Replicates CLAP4CLIP init_new_heads():
          mu_adapter.fc.weight    ← (X.T @ X) / ctx_dim
          sigma_adapter.fc.weight ← (X_var.T @ X_var) / ctx_dim
        where X = all current class features (normalised).
        """
        feats = self._frozen_text_feats_individual   # [K, n_tpl, 512]  (n_tpl≥1)
        if feats is None or feats.shape[0] == 0:
            return

        with torch.no_grad():
            if feats.dim() == 3:
                mean_feats = feats.mean(dim=1).float()                      # [K, ctx_dim]
                # unbiased=False avoids NaN when n_tpl=1 (Bessel correction 1/(n-1) → 1/0)
                var_feats  = feats.var(dim=1, unbiased=False).float()       # [K, ctx_dim]
            else:
                mean_feats = feats.float()               # [K, ctx_dim]
                var_feats  = torch.zeros_like(mean_feats)

            D = mean_feats.shape[1]
            layer_embed_mu    = (mean_feats.t() @ mean_feats) / D  # [D, D]
            layer_embed_sigma = (var_feats.t()  @ var_feats)  / D  # [D, D]

            mu_adp = self.mu_adapters[-1]
            mu_adp.fc.weight.data.copy_(
                layer_embed_mu.to(mu_adp.fc.weight.device).to(mu_adp.fc.weight.dtype)
            )
            sig_adp = self.sigma_adapters[-1]
            sig_adp.fc.weight.data.copy_(
                layer_embed_sigma.to(sig_adp.fc.weight.device).to(sig_adp.fc.weight.dtype)
            )

    def _forward_text_branch(
        self,
        img_clip: torch.Tensor,   # [B, ctx_dim=512] normalised CLIP image features
        n_tasks: int,
        train:   bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Full CLAP4CLIP-aligned text branch forward.

        Both image and text features are in 512-dim CLIP embed space.

        Steps:
          1. Build query = [text_frozen, task_tokens...]  [1, K+T, 512]
          2. Compute attention mask                       [K+T, K+T]
          3. Run shared VGA  (memory = img_clip)          vga_out [K+T, 512]
          4. For each task i:
               text_enh_i  = text_frozen_i + vga_class_i + vga_task_token_i
               qdist_i     = Normal(mu_i(text_enh_i), sigma_i(text_enh_i))
               samples_i  ~ qdist_i.rsample([forward_times])  [FT, n_cls_i, 512]
               text_fin_i  = text_enh_i + samples_i
               logits_i    = logit_scale * img_clip @ norm(text_fin_i).T  (mean FT)
          5. KL loss for current task only (during training)

        Args:
            img_clip : [B, 512] normalised CLIP image features (post visual.proj).
                       Passed as VGA memory (detached) and logit dot-product.
            n_tasks  : number of tasks seen so far.
            train    : True during training forward, False at eval.
        """
        text_f = self._frozen_text_feats.float()           # [K, 512]
        K = text_f.shape[0]

        # ── Build query ────────────────────────────────────────────────────────
        T = 0
        if self.use_task_tokens and self.task_tokens is not None:
            T = min(n_tasks, len(self.task_tokens))
            query = torch.cat(
                [text_f.unsqueeze(0)] + [self.task_tokens[i] for i in range(T)],
                dim=1,
            )                                               # [1, K+T, 512]
        else:
            query = text_f.unsqueeze(0)                    # [1, K, 512]

        # ── Attention mask ─────────────────────────────────────────────────────
        attn_mask = self._get_attention_mask(n_tasks, K)   # [K+T, K+T]

        # ── Shared VGA ─────────────────────────────────────────────────────────
        # CLAP4CLIP: context = image_features_normed [1, B, 512], same dim as query
        context = img_clip.detach().unsqueeze(0)           # [1, B, 512]
        if self.use_vga and self.vga is not None:
            vga_out = self.vga(query, context, tgt_mask=attn_mask).squeeze(0)  # [K+T, 512]
        else:
            vga_out = None

        # ── Per-task variational logits ────────────────────────────────────────
        logit_scale = self.clip_model.logit_scale.exp().clamp(max=100.0)

        current_task_idx = n_tasks - 1
        all_logits: List[torch.Tensor] = []
        kl_losses:  List[torch.Tensor] = []
        start_cls = 0

        for task_idx, n_cls in enumerate(self._cls_per_task[:n_tasks]):
            if task_idx >= len(self.mu_adapters):
                break
            end_cls = start_cls + n_cls

            # ── Enhanced text features (CLAP4CLIP: text + vga_class + vga_token) ─
            text_task = text_f[start_cls:end_cls].clone()          # [n_cls, 512]
            if vga_out is not None:
                text_task = text_task + vga_out[start_cls:end_cls]
                if self.use_task_tokens and T > task_idx:
                    # Broadcast task-token VGA residual to all classes of this task
                    text_task = text_task + vga_out[K + task_idx]

            # ── Variational adapter ────────────────────────────────────────────
            mu = self.mu_adapters[task_idx](text_task)             # [n_cls, 512]
            if self.use_variational and task_idx < len(self.sigma_adapters):
                sigma   = self.sigma_adapters[task_idx](text_task) # [n_cls, 512]
                dist    = Normal(mu, sigma)
                samples = dist.rsample([self.forward_times])        # [FT, n_cls, 512]
                if train and task_idx == current_task_idx and self.kl_loss_weight > 0.0:
                    prior = Normal(torch.zeros_like(mu), torch.ones_like(mu))
                    kl_losses.append(kl_divergence(dist, prior).mean(0).sum())
            else:
                samples = mu.unsqueeze(0)                           # [1, n_cls, 512]

            # ── Compute logits (all in 512-dim CLIP space, no extra projection) ─
            # text_sampled: text_task + samples  [FT, n_cls, 512]
            text_sampled = text_task.unsqueeze(0) + samples
            text_norm    = F.normalize(text_sampled.float(), dim=-1)  # [FT, n_cls, 512]

            # einsum("bd,tnd->tbn"): [B,512] × [FT,n_cls,512] → [FT,B,n_cls] → mean → [B,n_cls]
            logits_task = logit_scale * torch.einsum(
                "bd,tnd->tbn", img_clip.float(), text_norm
            )
            all_logits.append(logits_task.mean(0))
            start_cls = end_cls

        if not all_logits:
            return torch.zeros(img_clip.shape[0], 1, device=img_clip.device), None

        logits  = torch.cat(all_logits, dim=-1)                    # [B, K_total]
        kl_loss = sum(kl_losses) if kl_losses else None
        return logits, kl_loss

    # ──────────────────────────────────────────────────────────────────────────
    #  CLMethod interface
    # ──────────────────────────────────────────────────────────────────────────

    def adaptation(self, task_id: int, reset: bool = False) -> None:
        """
        Called once per session.  Sets up vision expert (super), then adds a
        new task token + variational adapters, refreshes text features, and
        initialises adapter heads from text feature covariance (CLAP4CLIP).
        """
        # Let FlyMethod handle visual expert, head expansion, RP head, etc.
        super().adaptation(task_id, reset=reset)

        # ── Count new classes ──────────────────────────────────────────────────
        n_total = self._num_seen_classes                  # updated by super()
        n_prev  = sum(self._cls_per_task)
        n_new   = max(1, n_total - n_prev)
        self._cls_per_task.append(n_new)

        # ── Refresh text features for ALL seen classes ─────────────────────────
        # Must happen BEFORE adapter init (init uses all-class covariance)
        self._refresh_text_features()

        # ── Task tokens (CLAP4CLIP expandable_tokens) ─────────────────────────
        if self.use_task_tokens:
            if self.task_tokens is None:
                self.task_tokens = nn.ParameterList()
            else:
                # Freeze all existing task tokens
                for tok in self.task_tokens:
                    tok.requires_grad_(False)
            # New trainable task token: trunc_normal_ init (CLAP4CLIP style)
            new_tok = torch.zeros(
                1, 1, self._ctx_dim, dtype=torch.float32, device=self.device
            )
            nn.init.trunc_normal_(new_tok, std=0.02)
            self.task_tokens.append(nn.Parameter(new_tok))

        # ── Freeze previous adapters ───────────────────────────────────────────
        for i in range(len(self.mu_adapters)):
            for p in self.mu_adapters[i].parameters():
                p.requires_grad_(False)
            for p in self.sigma_adapters[i].parameters():
                p.requires_grad_(False)

        # ── Add new trainable adapters for current task ────────────────────────
        mu    = _VarAdapter(self._ctx_dim, sigma=False).to(self.device).float()
        sigma = _VarAdapter(self._ctx_dim, sigma=True ).to(self.device).float()
        self.mu_adapters.append(mu)
        self.sigma_adapters.append(sigma)

        # ── Initialise new adapter heads from text covariance (CLAP4CLIP) ──────
        self._init_new_adapter_heads()

        # ── VGA is shared and ALWAYS trainable (CLAP4CLIP behaviour) ──────────
        if self.vga is not None:
            for p in self.vga.parameters():
                p.requires_grad_(True)

        # Reset KL cache
        self._pending_kl_loss = None

        self._aux_info.update({
            "method":          "fly_vga",
            "fly_mode":        self.fly_mode,
            "task_id":         int(self.current_task),
            "seen_classes":    int(self._num_seen_classes),
            "n_task_adapters": int(len(self.mu_adapters)),
            "n_task_tokens":   int(len(self.task_tokens)) if self.task_tokens else 0,
            "use_vga":         bool(self.use_vga),
            "use_task_tokens": bool(self.use_task_tokens),
        })

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        image:    torch.Tensor,
        test:     bool = False,
        all_test: bool = False,
    ) -> torch.Tensor:
        del all_test

        if not self._cls_per_task:
            raise RuntimeError("FlyVGAMethod.adaptation() must be called before forward().")

        n_tasks = len(self._cls_per_task)

        # ── Step 1: query features for RP routing (no grad) ───────────────────
        with torch.no_grad():
            q = self._extract_query_features(image)          # [B, transformer.width]

        # ══════════════════════════════════════════════════════════════════════
        #  TEST branch
        # ══════════════════════════════════════════════════════════════════════
        if test:
            with torch.no_grad():
                # Lazy RP update
                if self._rp_dirty:
                    self.rp_head.update()
                    self._rp_dirty = False

                # Expert selection via RP router
                seen_experts = max(1, min(self.current_task + 1, self.task_num))
                rp_logits  = self.rp_head(q)[:, :seen_experts]
                expert_ids = torch.argmax(rp_logits, dim=-1)

                # Visual features → project to CLIP embed space [B, 512]
                cls      = self._extract_expert_features(image, q, expert_ids, train=False)
                img_clip = F.normalize(
                    cls.float() @ self.clip_model.visual.proj.float(), dim=-1
                )                                                # [B, 512]

                # CLAP4CLIP-aligned text branch (shared VGA, no per-task routing)
                logits, _ = self._forward_text_branch(img_clip, n_tasks, train=False)

                route_entropy, route_top = self._routing_stats(expert_ids, seen_experts)
                self._aux_info.update({
                    "method":            "fly_vga",
                    "fly_mode":          self.fly_mode,
                    "task_id":           int(self.current_task),
                    "seen_experts":      int(seen_experts),
                    "route_entropy":     route_entropy,
                    "routed_expert_top": route_top,
                    "rp_samples_accum":  int(self._rp_collect_samples_total),
                })
                return logits[:, :self._num_seen_classes]

        # ══════════════════════════════════════════════════════════════════════
        #  TRAIN branch
        # ══════════════════════════════════════════════════════════════════════
        active_expert = max(0, min(self.current_task, self.task_num - 1))
        expert_ids    = torch.full(
            (image.size(0),), active_expert, device=image.device, dtype=torch.long
        )

        # Visual features → project to CLIP embed space [B, 512]
        cls      = self._extract_expert_features(image, q, expert_ids, train=True)
        img_clip = F.normalize(
            cls.float() @ self.clip_model.visual.proj.float(), dim=-1
        )                                                        # [B, 512]

        # CLAP4CLIP-aligned text branch (shared VGA + task tokens + variational)
        logits, kl_loss = self._forward_text_branch(img_clip, n_tasks, train=True)
        self._pending_kl_loss = kl_loss

        # Update RP head
        with torch.no_grad():
            rp_labels = torch.full(
                (q.size(0),), active_expert, device=q.device, dtype=torch.long
            )
            self.rp_head.collect(q, rp_labels)
            self._rp_collect_samples_total += int(q.size(0))
            self._rp_dirty = True

        self._aux_info.update({
            "method":           "fly_vga",
            "fly_mode":         self.fly_mode,
            "task_id":          int(self.current_task),
            "active_expert":    int(active_expert),
            "rp_samples_accum": int(self._rp_collect_samples_total),
        })
        return logits[:, :self._num_seen_classes]

    # ──────────────────────────────────────────────────────────────────────────

    def auxiliary_loss(self) -> Optional[torch.Tensor]:
        """Return KL divergence loss for the current-task adapter (if any)."""
        if self._pending_kl_loss is not None and self.kl_loss_weight > 0.0:
            return self.kl_loss_weight * self._pending_kl_loss
        return None

    def auxiliary_info(self) -> dict:
        info = super().auxiliary_info()
        info["method"]           = "fly_vga"
        info["use_vga"]          = bool(self.use_vga)
        info["use_task_tokens"]  = bool(self.use_task_tokens)
        info["use_variational"]  = bool(self.use_variational)
        info["forward_times"]    = int(self.forward_times)
        info["kl_loss_weight"]   = float(self.kl_loss_weight)
        info["kl_loss"] = (
            float(self._pending_kl_loss.item() * self.kl_loss_weight)
            if self._pending_kl_loss is not None
            else 0.0
        )
        return info

