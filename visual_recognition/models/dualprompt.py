import logging
from typing import Iterable

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit
from models.l2p import Prompt
from models.flyprompt import RPFC


def _stable_cholesky(matrix: torch.Tensor, reg: float = 1e-4) -> torch.Tensor:
    eye = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    return torch.linalg.cholesky(matrix + reg * eye)


def _transform_to_target_covariance(features: torch.Tensor,
                                     target_cov: torch.Tensor,
                                     reg: float = 1e-4) -> torch.Tensor:
    """Align feature covariance to target_cov using a linear transform.

    features: [B, D]
    target_cov: [D, D]
    """
    if features.size(0) <= 1:
        # Not enough samples to estimate covariance; skip calibration.
        return features

    orig_dtype = features.dtype
    # Compute covariance and transform in float32 for numerical stability
    features_f = features.to(dtype=torch.float32)
    target_cov_f = target_cov.to(device=features.device, dtype=torch.float32)

    centered = features_f - features_f.mean(dim=0, keepdim=True)
    n = centered.size(0)
    C = centered.T @ centered / (n - 1)
    L = _stable_cholesky(C, reg)
    L_target = _stable_cholesky(target_cov_f, reg)
    A = torch.linalg.solve(L, L_target)
    Fj = centered @ A
    return Fj.to(dtype=orig_dtype)


logger = logging.getLogger()


class DualPrompt(nn.Module):
    def __init__(self,
                 pos_g_prompt   : Iterable[int] = (0, 1),
                 len_g_prompt   : int   = 5,
                 pos_e_prompt   : Iterable[int] = (2,3,4),
                 len_e_prompt   : int   = 20,
                 g_pool         : int   = 1,
                 e_pool         : int   = 10,
                 prompt_func    : str   = 'prompt_tuning',
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 lambd          : float = 1.0,
                 backbone_name  : str   = None,
                 load_pt        : bool  = False,
                 **kwargs):
        super().__init__()

        self.lambd = lambd
        self.kwargs = kwargs
        self.task_num = task_num
        self.num_classes = num_classes

        # MePo configuration
        self.mepo_backbone_path = self.kwargs.get("mepo_backbone_path", None)
        self.cov_path = self.kwargs.get("cov_path", None)
        self.cov_coef = float(self.kwargs.get("cov_coef", 0.7))
        self.cov_coef = max(0.0, min(1.0, self.cov_coef)) # Enforce cov_coef in [0, 1]

        # Require both MePo paths to be specified together, or neither
        if (self.mepo_backbone_path is None) != (self.cov_path is None):
            raise ValueError(
                "For MePo, both mepo_backbone_path and cov_path must be provided; "
                "set both or leave both as None for plain DualPrompt."
            )

        # Effective expert-prompt pool size.
        # - For step_num/task_num <= 10, we always use e_pool = 10.
        # - For step_num/task_num == 20, we use e_pool = 20.
        # - For other values, fall back to the provided e_pool (but ensure e_pool >= task_num).
        if self.task_num <= 10:
            e_pool = 10
        elif self.task_num == 20:
            e_pool = 20
        else:
            if e_pool < self.task_num:
                e_pool = self.task_num

        self.task_count = 0

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        # Use custom ViT model from models.vit to support local .npz loading
        if hasattr(vit, backbone_name):
            logger.info(f'Using custom ViT model: {backbone_name}')
            self.add_module('backbone', getattr(vit, backbone_name)(pretrained=True, num_classes=num_classes))
        else:
            logger.info(f'Using timm model: {backbone_name}')
            self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))

        # Optionally override backbone weights with MePo checkpoint (without loading fc/head)
        if self.mepo_backbone_path is not None:
            logger.info(f"Loading MePo backbone from {self.mepo_backbone_path}")
            self._load_mepo_backbone(self.mepo_backbone_path)

        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True

        # Optionally load covariance matrix for MePo calibration
        if self.cov_path is not None:
            self._load_mepo_covariance(self.cov_path)

        # RPFC-based task gating (optional, FlyPrompt-style)
        self.use_rp_gate = kwargs.get("use_rp_gate", False)
        rp_dim = kwargs.get("rp_dim", 0)
        rp_ridge = kwargs.get("rp_ridge", 1e4)
        if self.use_rp_gate:
            self.rp_head = RPFC(
                M=rp_dim,
                ridge=rp_ridge,
                embed_dim=self.backbone.num_features,
                num_classes=task_num,
            )
        else:
            self.rp_head = None

        # EMA-based classifier head bank (optional, per-prompt-slot experts)
        self.use_ema_head = kwargs.get("use_ema_head", False)
        ema_ratio = kwargs.get("ema_ratio", [0.9, 0.99])
        if isinstance(ema_ratio, (float, int)):
            ema_ratio = [float(ema_ratio)]
        self.ema_ratio = [float(r) for r in ema_ratio]
        self.num_ema = len(self.ema_ratio) if self.use_ema_head else 0
        self.last_expert_ids = None

        if self.use_ema_head and self.num_ema > 0:
            # One EMA head bank per prompt slot in the expert pool
            self.experts_fc = nn.ModuleList(
                [
                    nn.ModuleList(
                        [nn.Linear(self.backbone.num_features, self.num_classes, bias=True) for _ in range(self.num_ema)]
                    )
                    for _ in range(e_pool)
                ]
            )
            for expert_fc in self.experts_fc:
                for fc in expert_fc:
                    for param in fc.parameters():
                        param.requires_grad = False
            # Initialize EMA heads from the online classifier
            self.init_fc()
        else:
            self.experts_fc = None

        # Slice the eprompt. Ensure at least one prompt slot per task/step.
        if task_num <= 0:
            raise ValueError(f"task_num must be positive, got {task_num}")
        self.num_pt_per_task = max(1, int(e_pool / task_num))

        self.e_pool = e_pool
        self.len_g_prompt = len_g_prompt if not load_pt else 10
        self.len_e_prompt = len_e_prompt
        self.g_length = len(pos_g_prompt) if pos_g_prompt else 0
        self.e_length = len(pos_e_prompt) if pos_e_prompt else 0

        self.register_buffer('pos_g_prompt', torch.tensor(pos_g_prompt, dtype=torch.int64))
        self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype=torch.int64))
        self.register_buffer('similarity', torch.ones(1).view(1))

        if prompt_func == 'prompt_tuning':
            self.prompt_func = self.prompt_tuning
            self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(
                g_pool, 1, self.g_length * self.len_g_prompt, self.backbone.num_features,
                _batchwise_selection=False, _diversed_selection=False, kwargs=self.kwargs
                )
            self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(
                e_pool, 1, self.e_length * self.len_e_prompt, self.backbone.num_features,
                _batchwise_selection=False, _diversed_selection=False, kwargs=self.kwargs
                )
        elif prompt_func == 'prefix_tuning':
            self.prompt_func = self.prefix_tuning
            self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(
                g_pool, 1, 2 * self.g_length * self.len_g_prompt, self.backbone.num_features,
                _batchwise_selection=False, _diversed_selection=False, kwargs=self.kwargs
                )
            self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(
                e_pool, 1, 2 * self.e_length * self.len_e_prompt, self.backbone.num_features,
                _batchwise_selection=False, _diversed_selection=False, kwargs=self.kwargs
                )
        else: raise ValueError('Unknown prompt_func: {}'.format(prompt_func))
        self.g_prompt.key = None

        self.load_prompt(load_pt)

    def load_prompt(self, load_pt: bool = False,):
        g_path = "./checkpoints/g_prompt.pt"
        e_path = "./checkpoints/e_prompt.pt"
        if load_pt:
            logger.info(f"load prompt from {g_path} and {e_path}")
            g_prompt = torch.load(g_path)
            e_prompt = torch.load(e_path)
            self.g_prompt.prompts = nn.Parameter(g_prompt.detach().clone())

            # e_prompt in checkpoints is assumed to have pool size 10. When the
            # current model uses a larger expert pool (e.g., e_pool=20 for
            # step_num/task_num=20), tile the loaded prompts along the pool
            # dimension so that shapes match.
            e_prompt = e_prompt.detach()
            if e_prompt.dim() == 3:
                old_e_pool = e_prompt.size(0)
                new_e_pool = self.e_pool
                if old_e_pool != new_e_pool:
                    # Tile along pool dimension until reaching new_e_pool
                    repeat_factor = (new_e_pool + old_e_pool - 1) // old_e_pool
                    e_prompt = e_prompt.repeat(repeat_factor, 1, 1)[:new_e_pool]
            self.e_prompt.prompts = nn.Parameter(e_prompt.clone())

    def prompt_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, self.e_length, self.len_e_prompt, C)
        g_prompt = g_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.e_length, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                x = torch.cat((x, g_prompt[:, pos_g]), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                x = torch.cat((x, e_prompt[:, pos_e]), dim = 1)
            x = block(x)
            x = x[:, :N, :]
        return x

    def prefix_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, 2 * self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, 2 * self.e_length, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            xq = block.norm1(x)
            xk = xq.clone()
            xv = xq.clone()

            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                xk = torch.cat(xk, (g_prompt[:, pos_g * 2 + 0]), dim = 1)
                xv = torch.cat(xv, (g_prompt[:, pos_g * 2 + 1]), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                xk = torch.cat(xk, (e_prompt[:, pos_e * 2 + 0]), dim = 1)
                xv = torch.cat(xv, (e_prompt[:, pos_e * 2 + 1]), dim = 1)

            attn   = block.attn
            weight = attn.qkv.weight
            bias   = attn.qkv.bias

            B, N, C = xq.shape
            xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xk.shape
            xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xv.shape
            xv = F.linear(xv, weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)

            attention = (xq @ xk.transpose(-2, -1)) * attn.scale
            attention = attention.softmax(dim=-1)
            attention = attn.attn_drop(attention)

            attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
            attention = attn.proj(attention)
            attention = attn.proj_drop(attention)

            x = x + block.drop_path1(block.ls1(attention))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

        return x

    def forward(self, inputs : torch.Tensor, return_feat=False) :
        with torch.no_grad():
            x = self.backbone.patch_embed(inputs)
            B, N, D = x.size()

            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0]

        if self.g_prompt is not None:
            g_p = self.g_prompt.prompts[0]
            g_p = g_p.expand(B, -1, -1)
        else:
            g_p = None

        if self.e_prompt is not None:
            start_id = self.task_count * self.num_pt_per_task
            end_id = (self.task_count+1) * self.num_pt_per_task
            if self.training and start_id < self.e_pool:
                res_e = self.e_prompt(query, s=start_id, e=end_id)
            else:
                if getattr(self, "use_rp_gate", False) and (not self.training) and (self.rp_head is not None):
                    res_e = self._select_e_prompt_with_rp(query)
                else:
                    res_e = self.e_prompt(query)
            e_s, e_p = res_e

            # Record last expert (prompt slot) indices for EMA experts when routing via Prompt
            if not (getattr(self, "use_rp_gate", False) and (not self.training) and (self.rp_head is not None)):
                if hasattr(self.e_prompt, "last_selected_indices") and self.e_prompt.last_selected_indices is not None:
                    # Use top-1 selected prompt as the expert id.
                    self.last_expert_ids = self.e_prompt.last_selected_indices[:, 0].detach()
        else:
            e_p = None
            e_s = 0
            self.last_expert_ids = None

        x = self.prompt_func(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_p, e_p)
        x = self.backbone.norm(x)
        cls_token = x[:, 0]

        # Apply MePo covariance calibration only after prompts and transformer
        cls_token = self._apply_mepo_cov_calibration(cls_token)

        x = self.backbone.fc(cls_token)

        # keep similarity for compatibility
        if isinstance(e_s, torch.Tensor):
            self.similarity = e_s.mean()
        else:
            self.similarity = torch.tensor(0., device=x.device)

        if return_feat:
            return x, cls_token
        else:
            return x

    def _apply_mepo_cov_calibration(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Apply MEPO covariance calibration to CLS token if enabled.

        This uses the batch CLS features to estimate current covariance and
        aligns it to the target covariance matrix loaded from cov_path.
        """
        if getattr(self, "cov_matrix", None) is None:
            return cls_token

        # Run MEPO calibration in full precision regardless of outer AMP context
        with torch.cuda.amp.autocast(enabled=False):
            cls_fp32 = cls_token.to(dtype=torch.float32)
            cov = self.cov_matrix.to(device=cls_fp32.device, dtype=torch.float32)
            Fj = _transform_to_target_covariance(cls_fp32, cov)
            # Normalize to unit norm to avoid scale explosion
            norm = Fj.norm(dim=1, keepdim=True).clamp_min(1e-6)
            Fj = Fj / norm

            out = (1.0 - float(self.cov_coef)) * cls_fp32 + float(self.cov_coef) * Fj

        return out.to(dtype=cls_token.dtype)

    def _load_mepo_backbone(self, ckpt_path: str) -> None:
        """Load MEPO backbone weights from a checkpoint without loading fc/head.

        The provided meta_epoch_*.pth checkpoints are plain state_dicts with
        ViT backbone weights (cls_token, pos_embed, patch_embed, blocks, norm).
        We also defensively drop any keys that look like classifier heads.
        """
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            state_dict = state
        else:
            state_dict = state

        new_state_dict = {}
        for k, v in state_dict.items():
            # Strip common wrappers if ever present
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            # Do not load classifier heads
            if k.startswith("fc.") or k.startswith("head."):
                continue
            new_state_dict[k] = v

        missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
        if missing:
            logger.warning(f"[MEPO] Missing keys when loading backbone from {ckpt_path}: {missing}")
        if unexpected:
            logger.warning(f"[MEPO] Unexpected keys when loading backbone from {ckpt_path}: {unexpected}")

    def _load_mepo_covariance(self, cov_path: str) -> None:
        """Load covariance matrix from .npy and register as buffer."""
        cov = np.load(cov_path)
        cov = torch.from_numpy(cov).float()
        if cov.dim() != 2 or cov.size(0) != cov.size(1):
            raise ValueError(f"Covariance matrix at {cov_path} must be square, got {cov.shape}")
        if hasattr(self.backbone, "num_features"):
            feat_dim = self.backbone.num_features
        else:
            # Fallback: infer from cls_token dimension at runtime
            feat_dim = cov.size(0)
        if cov.size(0) != feat_dim:
            raise ValueError(
                f"Covariance dim {cov.size(0)} does not match backbone features {feat_dim}"
            )
        self.register_buffer("cov_matrix", cov)
        logger.info(f"[MEPO] Loaded covariance matrix from {cov_path} with shape {cov.shape}.")

    def forward_with_ema(self, inputs: torch.Tensor):
        """Forward with online head + EMA heads.

        This is used only during evaluation. It runs the standard forward
        once to obtain classifier logits and CLS features, then applies the
        EMA expert heads based on the recorded prompt-slot expert ids.
        Returns a list of logits: [online, ema_1, ema_2, ...].
        """
        logits, cls_token = self.forward(inputs, return_feat=True)
        outputs_ls = [logits]

        if not self.use_ema_head or self.experts_fc is None or self.num_ema == 0:
            return outputs_ls

        expert_ids = getattr(self, "last_expert_ids", None)
        if expert_ids is None:
            return outputs_ls

        expert_ids = expert_ids.to(cls_token.device).long()
        for i in range(self.num_ema):
            outputs = []
            for feat_i, e_i in zip(cls_token, expert_ids):
                e_idx = int(e_i.item())
                if e_idx < 0 or e_idx >= self.e_pool:
                    e_idx = max(0, min(self.e_pool - 1, e_idx))
                outputs.append(self.experts_fc[e_idx][i](feat_i))
            outputs_ls.append(torch.stack(outputs, dim=0))

        return outputs_ls

    @torch.no_grad()
    def init_fc(self, expert_ids: torch.Tensor = None):
        """Initialize EMA classifier heads from the online classifier.

        If ``expert_ids`` is None, initialize all experts; otherwise only the
        specified prompt-slot indices are initialized. This allows newly
        activated prompts (for later tasks) to start from the current online
        classifier weights.
        """
        if not self.use_ema_head or self.experts_fc is None or self.num_ema == 0:
            return

        src_weight = self.backbone.fc.weight.data
        src_bias = self.backbone.fc.bias.data

        if expert_ids is None:
            indices = range(len(self.experts_fc))
        else:
            indices = [int(i) for i in expert_ids.detach().cpu().tolist()]

        for e_idx in indices:
            if e_idx < 0 or e_idx >= len(self.experts_fc):
                continue
            expert_fc = self.experts_fc[e_idx]
            for fc in expert_fc:
                fc.weight.data.copy_(src_weight)
                fc.bias.data.copy_(src_bias)

    @torch.no_grad()
    def update_ema_fc(self, expert_ids: torch.Tensor):
        """Update EMA classifier heads for the given expert (prompt-slot) ids.

        Args:
            expert_ids: 1D tensor of prompt-slot indices used in the batch.
        """
        if not self.use_ema_head or self.experts_fc is None or self.num_ema == 0:
            return
        if expert_ids is None or expert_ids.numel() == 0:
            return

        # Use unique ids to avoid redundant updates
        unique_ids = expert_ids.detach().to(self.backbone.fc.weight.device).long().unique()
        for e_idx in unique_ids.tolist():
            if e_idx < 0 or e_idx >= self.e_pool:
                continue
            for i, ema_ratio in enumerate(self.ema_ratio):
                ema_fc = self.experts_fc[e_idx][i]
                ema_fc.weight.data.mul_(ema_ratio).add_(self.backbone.fc.weight.data, alpha=1.0 - ema_ratio)
                ema_fc.bias.data.mul_(ema_ratio).add_(self.backbone.fc.bias.data, alpha=1.0 - ema_ratio)


    def _select_e_prompt_with_rp(self, query: torch.Tensor):
        """Select e-prompts using RPFC task gating.

        Args:
            query: CLS features from backbone, shape [B, D].
        Returns:
            (similarity, prompts) same format as Prompt.forward.
        """
        if self.rp_head is None or self.e_prompt is None:
            res_e = self.e_prompt(query)
            # Fallback: record expert ids directly from Prompt routing
            if hasattr(self.e_prompt, "last_selected_indices") and self.e_prompt.last_selected_indices is not None:
                self.last_expert_ids = self.e_prompt.last_selected_indices[:, 0].detach()
            return res_e

        B, D = query.shape
        device = query.device

        # Predict task ids via RPFC; only consider seen tasks.
        E = min(self.task_count + 1, self.task_num)
        logits = self.rp_head(query)
        logits = logits[:, :E]
        task_hat = torch.argmax(logits, dim=-1)  # [B]

        selection_size = self.e_prompt.selection_size
        prompt_len = self.e_prompt.prompt_len
        dim = self.e_prompt.dimention

        e_s = torch.zeros(B, selection_size, device=device, dtype=query.dtype)
        e_p = torch.zeros(B, selection_size, prompt_len, dim, device=device, dtype=query.dtype)
        expert_ids = torch.zeros(B, device=device, dtype=torch.long)

        for t in range(E):
            idx = (task_hat == t).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            q_t = query[idx]
            start_id = t * self.num_pt_per_task
            end_id = min((t + 1) * self.num_pt_per_task, self.e_pool)
            if start_id >= self.e_pool:
                res_e = self.e_prompt(q_t)
            else:
                res_e = self.e_prompt(q_t, s=start_id, e=end_id)
            e_s_t, e_p_t = res_e
            e_s[idx] = e_s_t
            e_p[idx] = e_p_t

            # Record expert ids for this subset using Prompt's cached indices
            if hasattr(self.e_prompt, "last_selected_indices") and self.e_prompt.last_selected_indices is not None:
                local_ids = self.e_prompt.last_selected_indices[:, 0].detach()
                if local_ids.shape[0] == idx.shape[0]:
                    expert_ids[idx] = local_ids.to(device)

        self.last_expert_ids = expert_ids
        return e_s, e_p

    def get_e_prompt_count(self):
        return self.e_prompt.update()

    def process_task_count(self):
        self.task_count += 1
        # Initialize EMA heads for newly activated prompt slots of the next task
        if self.use_ema_head and self.experts_fc is not None and self.num_ema > 0:
            start_id = self.task_count * self.num_pt_per_task
            end_id = min((self.task_count + 1) * self.num_pt_per_task, self.e_pool)
            if start_id < self.e_pool:
                ids = torch.arange(start_id, end_id, device=self.backbone.fc.weight.device, dtype=torch.long)
                self.init_fc(expert_ids=ids)

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target) + self.lambd * self.similarity