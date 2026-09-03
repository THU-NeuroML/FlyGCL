"""
PROOF method — fully aligned with PROOF-main/models/proof.py and
PROOF-main/utils/inc_net.py::Proof_Net.

Key design decisions (matching the original):
  - Main training loop (main.py epochs) is a no-op for PROOF: forward() in
    train mode returns raw CLIP zero-shot logits so the optimizer only touches
    parameters that have requires_grad=True (none during the main loop phase).
    All real training happens inside after_task() -> _train_proj().
  - encode_image / encode_text apply ALL task projection heads (sum), matching
    Proof_Net.encode_image / encode_text.
  - Prototypes are mean of *normalized* raw CLIP features (cal_prototype uses
    encode_image(normalize=True) which is raw CLIP + normalize, no projection).
  - _train_proj freezing: all projs_img frozen except last; all projs_text
    unfrozen; sel_attn unfrozen; logit_scale unfrozen.
  - Inference: original_outputs + transf_outputs + proto_outputs (three-way sum,
    matching _compute_accuracy).
  - clip_loss uses projected+normalized img_feas (not raw CLIP), matching
    proof.py line 146: img_feas = encode_image(inputs) / norm.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import json
import os
import logging

import open_clip
import open_clip.openai as openai_clip
from open_clip.transform import image_transform
from ..utils import get_class_ids_per_task, get_class_names
from .base import CLMethod


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        attn = self.dropout(attn.softmax(dim=2))
        return torch.bmm(attn, v)


class _MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head, self.d_k, self.d_v = n_head, d_k, d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = _ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v
        sz_b, len_q, _ = q.size()
        residual = q
        q  = self.w_qs(q).view(sz_b, len_q, n_head, d_k).permute(2,0,1,3).contiguous().view(-1, len_q, d_k)
        k  = self.w_ks(k).view(sz_b, k.size(1), n_head, d_k).permute(2,0,1,3).contiguous().view(-1, k.size(1), d_k)
        v  = self.w_vs(v).view(sz_b, v.size(1), n_head, d_v).permute(2,0,1,3).contiguous().view(-1, v.size(1), d_v)
        output = self.attention(q, k, v)
        output = output.view(n_head, sz_b, len_q, d_v).permute(1,2,0,3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        return self.layer_norm(output + residual)


class _Proj_Pure_MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.mlp(x)


class PROOFMethod(CLMethod):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.classes_names = None
        self._label_list = None
        self.prompt_template = cfg.prompt_template
        self.prompt_templates = list(cfg.get("prompt_templates", [cfg.prompt_template]))
        self._train_prompt_template = self.prompt_templates[0]

        model_name = cfg.get("proof_model_name", "ViT-B-16")
        pretrained = cfg.get("proof_pretrained", "laion400m_e32")
        local_pretrained = cfg.get(
            "proof_pretrained_path",
            "",
        )
        use_local_pretrained = bool(local_pretrained and os.path.isfile(local_pretrained))
        if local_pretrained and os.path.isfile(local_pretrained):
            logging.info(f"[PROOF] using local pretrained checkpoint: {local_pretrained}")
            pretrained = local_pretrained
        else:
            logging.info(f"[PROOF] local checkpoint not found, fallback to pretrained tag: {pretrained}")

        # PyTorch>=2.6 sets torch.load(weights_only=True) by default, which fails
        # when local CLIP checkpoints are TorchScript archives.
        orig_torch_load = torch.load
        if use_local_pretrained:
            def _torch_load_compat(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return orig_torch_load(*args, **kwargs)
            torch.load = _torch_load_compat

        try:
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
        except AttributeError as e:
            # Local OpenAI CLIP TorchScript checkpoint is not a state_dict; load via openai branch.
            if use_local_pretrained and "items" in str(e):
                logging.info("[PROOF] fallback: loading local TorchScript checkpoint via open_clip.openai")
                clip_model = openai_clip.load_openai_model(local_pretrained, device=device, jit=False)
                image_size = clip_model.visual.image_size
                if isinstance(image_size, (tuple, list)):
                    image_size = image_size[-1]
                preprocess = image_transform(image_size=int(image_size), is_train=False)
            else:
                raise
        finally:
            if use_local_pretrained:
                torch.load = orig_torch_load
        if hasattr(open_clip, "get_tokenizer"):
            self.tokenizer = open_clip.get_tokenizer(model_name)
        elif hasattr(open_clip, "tokenize"):
            self.tokenizer = open_clip.tokenize
        else:
            raise RuntimeError("open_clip tokenizer API not found (neither get_tokenizer nor tokenize)")
        clip_model = clip_model.to(device).float()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = clip_model
        self.transforms = preprocess

        feat_dim = clip_model.text_projection.shape[1]  # 512 for ViT-B/16
        self._feat_dim = feat_dim

        self.projs_img  = nn.ModuleList()
        self.projs_text = nn.ModuleList()
        self.sel_attn   = _MultiHeadAttention(1, feat_dim, feat_dim, feat_dim, dropout=0.1).to(device)
        self.context_prompts = nn.ParameterList()
        self.context_prompt_length = cfg.get("context_prompt_length_per_task", 3)

        # img_prototypes[i]: mean of *normalized* raw CLIP features for class i
        self.img_prototypes = None

        self.class_ids_per_task  = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.current_class_ids   = []
        self.text_tokens         = None
        self.current_task        = -1
        self._known_classes      = 0
        self._total_classes      = 0

        self._proj_epochs = cfg.get("proof_proj_epochs", 5)
        self._proj_lr     = cfg.get("proof_proj_lr", 0.001)
        self._min_lr      = cfg.get("proof_min_lr", 1e-8)
        self._proj_weight_decay = float(cfg.get("weight_decay", 0.0))
        self._proj_optimizer = str(cfg.get("proof_optimizer", "sgd")).lower()
        self._proj_momentum = float(cfg.get("momentum", 0.9))
        self._proof_log_batch_interval = int(cfg.get("log_batch_interval", 10))
        self._objective_mode = str(cfg.get("proof_objective_mode", "paper_eq10")).lower()
        if self._objective_mode == "paper_interpreted_three_branch":
            # Backward-compatible alias used by older configs.
            self._objective_mode = "paper_eq10"
        if self._objective_mode not in {"repo_exact", "paper_eq10"}:
            logging.warning(
                f"[PROOF] unknown proof_objective_mode='{self._objective_mode}', fallback to paper_eq10"
            )
            self._objective_mode = "paper_eq10"

        # GCL fairness controls for PROOF task-end projection stage.
        # full: keep original PROOF-style projection training.
        # lightweight: run a reduced projection stage for fairer per-session compute.
        # disabled: skip projection stage entirely (prototype update still runs).
        self._projection_enable = bool(cfg.get("proof_projection_enable", True))
        self._projection_mode = str(cfg.get("proof_projection_mode", "lightweight")).lower()
        self._proj_light_epochs = int(cfg.get("proof_proj_light_epochs", 1))
        self._proj_max_steps = int(cfg.get("proof_proj_max_steps", -1))
        self._proj_light_max_steps = int(cfg.get("proof_proj_light_max_steps", -1))

        self._last_projection_info = {
            "enabled": False,
            "mode": "disabled",
            "epochs": 0,
            "max_steps_per_epoch": 0,
            "estimated_extra_steps": 0,
            "actual_extra_steps": 0,
            "trainable_tensors": 0,
            "trainable_params": 0,
            "objective_mode": self._objective_mode,
        }

        self._load_original_prompt_and_labels()

    def _proof_dataset_key(self):
        key = self.cfg.get("proof_dataset_key", None)
        if key is not None:
            return key
        dataset_name = str(getattr(self.cfg, "dataset", "")).lower()
        mapping = {
            "cifar100": "cifar224",
            "imagenet_R": "imagenetr",
            "imagenet_r": "imagenetr",
            "imagenet-r": "imagenetr",
        }
        return mapping.get(dataset_name, dataset_name)

    def _load_original_prompt_and_labels(self):
        root = os.path.join(
            os.path.dirname(__file__),
            "../../external/PROOF-main/utils"
        )
        dataset_key = self._proof_dataset_key()
        labels_path = os.path.join(root, "labels.json")
        templates_path = os.path.join(root, "templates.json")

        try:
            with open(templates_path, "r") as f:
                templates_db = json.load(f)
            if dataset_key in templates_db:
                self.prompt_templates = list(templates_db[dataset_key])
                self._train_prompt_template = self.prompt_templates[0]
        except Exception as e:
            logging.warning(f"[PROOF] template injection fallback to cfg prompt_templates: {e}")

        try:
            with open(labels_path, "r") as f:
                labels_db = json.load(f)
            if dataset_key in labels_db:
                self._label_list = labels_db[dataset_key]
        except Exception as e:
            logging.warning(f"[PROOF] labels injection fallback to dataset class names: {e}")

    def _resolve_task_class_names(self, class_ids):
        if self._label_list is not None:
            return [self._label_list[idx] for idx in class_ids]
        return get_class_names(self.classes_names, class_ids)

    # ------------------------------------------------------------------
    # CLMethod interface
    # ------------------------------------------------------------------

    def adaptation(self, task_id: int, reset: bool = False) -> None:
        if reset:
            logging.info("[PROOF] `reset` argument is ignored in PROOF path (kept for interface compatibility)")
        self.current_task += 1
        task_class_ids = [int(x) for x in self.class_ids_per_task[task_id]]
        task_class_names = self._resolve_task_class_names(task_class_ids)
        self.current_class_names += task_class_names
        self.current_class_ids += task_class_ids
        n_new = len(task_class_names)
        self._known_classes  = self._total_classes
        self._total_classes += n_new

        # Extend prototype buffer (zeros for new classes; filled in after_task)
        if self.img_prototypes is not None:
            self.img_prototypes = torch.cat([
                self.img_prototypes.detach(),
                torch.zeros(n_new, self._feat_dim, device=self.device)
            ], dim=0)
        else:
            self.img_prototypes = torch.zeros(
                self._total_classes, self._feat_dim, device=self.device
            )

        # New projection heads — frozen until _train_proj unfreezes them
        new_proj_img  = _Proj_Pure_MLP(self._feat_dim, self._feat_dim).to(self.device)
        new_proj_text = _Proj_Pure_MLP(self._feat_dim, self._feat_dim).to(self.device)
        for p in new_proj_img.parameters():
            p.requires_grad = False
        for p in new_proj_text.parameters():
            p.requires_grad = False
        self.projs_img.append(new_proj_img)
        self.projs_text.append(new_proj_text)

        # Freeze old context prompts, add new one (also frozen until _train_proj)
        for p in self.context_prompts:
            p.requires_grad = False
        new_ctx = nn.Parameter(
            torch.randn(self.context_prompt_length, self._feat_dim, device=self.device),
            requires_grad=False
        )
        self.context_prompts.append(new_ctx)

        # sel_attn frozen until _train_proj
        for p in self.sel_attn.parameters():
            p.requires_grad = False

        # Text tokens for all seen classes
        self.text_tokens = self.tokenizer(
            [self._train_prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        logging.info(f"[PROOF] task {task_id} prototype slots: {self._total_classes}, context prompts: {len(self.context_prompts)}")

    def forward(self, image: torch.Tensor, test: bool = False, all_test: bool = False) -> torch.Tensor:
        """
        Train mode: returns raw CLIP zero-shot logits (no projection).
        The main training loop in main.py is a no-op for PROOF — no projection
        parameters are trainable at this stage, so backward() updates nothing.
        All real training is done in after_task() -> _train_proj().

        Test mode: three-way logit sum matching proof.py::_compute_accuracy:
            outputs = original_outputs + transf_outputs + proto_outputs
        where all features are projected (encode_image / encode_text).
        """
        raw_img = self.clip_model.encode_image(image)

        if all_test:
            logging.info("[PROOF] `all_test` is handled by eval loader slicing in main; forward() uses seen classes.")

        if not test:
            raw_txt = self.clip_model.encode_text(self.text_tokens)
            raw_img_n = raw_img / raw_img.norm(dim=-1, keepdim=True)
            raw_txt_n = raw_txt / raw_txt.norm(dim=-1, keepdim=True)
            # No-op training pass: zero-shot CLIP logits, no projection params touched
            logit_scale = self.clip_model.logit_scale.exp()
            return logit_scale * raw_img_n @ raw_txt_n.t()

        # --- Test path ---
        # Match PROOF eval: image features from projected image encoder (unnormalized).
        img_proj = self._encode_image_proj(raw_img)   # (B, d)

        # Match PROOF eval text path: multi-template encode -> normalize -> mean -> normalize.
        txt_proj_list = []
        for class_name in self.current_class_names:
            class_tokens = self.tokenizer(
                [t.format(class_name) for t in self.prompt_templates]
            ).to(self.device)
            raw_txt_multi = self.clip_model.encode_text(class_tokens)
            txt_multi = self._encode_text_proj(raw_txt_multi)
            txt_multi = txt_multi / txt_multi.norm(dim=-1, keepdim=True)
            txt_cls = txt_multi.mean(dim=0)
            txt_cls = txt_cls / txt_cls.norm(dim=-1, keepdim=True)
            txt_proj_list.append(txt_cls)
        txt_proj = torch.stack(txt_proj_list, dim=0)  # (n_cls, d)

        proto_feat = self._encode_prototypes()        # (n_cls, d)

        context = torch.cat(list(self.context_prompts), dim=0)
        img_fused, txt_fused, proto_fused = self._forward_transformer(
            img_proj, txt_proj, proto_feat, context
        )

        # Three-way sum: original + transf + proto (proof.py _compute_accuracy)
        original_outputs = img_proj @ txt_proj.t()
        transf_outputs   = img_fused @ txt_fused.t()
        proto_outputs    = img_fused @ proto_fused.t()
        logits = original_outputs + transf_outputs + proto_outputs
        return logits

    def after_task(self, train_loader=None) -> None:
        """
        Mirrors proof.py::incremental_train order:
          1. cal_prototype  (compute prototypes for new classes)
          2. _train_proj    (fine-tune projection heads)
        """
        if train_loader is None:
            return
        self._cal_prototype(train_loader)

        plan = self._resolve_projection_plan(train_loader)
        self._last_projection_info = {
            "enabled": bool(plan["enabled"]),
            "mode": str(plan["mode"]),
            "epochs": int(plan["epochs"]),
            "max_steps_per_epoch": int(plan["max_steps_per_epoch"]),
            "estimated_extra_steps": int(plan["estimated_extra_steps"]),
            "actual_extra_steps": 0,
            "trainable_tensors": 0,
            "trainable_params": 0,
            "objective_mode": self._objective_mode,
        }
        logging.info(
            f"[PROOF] projection plan | enabled={int(plan['enabled'])} mode={plan['mode']} "
            f"epochs={plan['epochs']} max_steps_per_epoch={plan['max_steps_per_epoch']} "
            f"estimated_extra_steps={plan['estimated_extra_steps']} objective_mode={self._objective_mode}"
        )

        if not plan["enabled"] or plan["epochs"] <= 0:
            logging.info("[PROOF] projection stage skipped by config")
            return

        stats = self._train_proj(
            train_loader,
            epochs=int(plan["epochs"]),
            max_steps_per_epoch=int(plan["max_steps_per_epoch"]),
            mode=str(plan["mode"]),
        )
        self._last_projection_info.update(stats)

    def projection_info(self):
        return dict(self._last_projection_info)

    def _resolve_projection_plan(self, train_loader):
        mode = self._projection_mode
        if mode not in {"full", "lightweight", "disabled"}:
            logging.warning(f"[PROOF] unknown proof_projection_mode='{mode}', fallback to lightweight")
            mode = "lightweight"

        if not self._projection_enable or mode == "disabled":
            return {
                "enabled": False,
                "mode": "disabled",
                "epochs": 0,
                "max_steps_per_epoch": 0,
                "estimated_extra_steps": 0,
            }

        if mode == "full":
            epochs = max(int(self._proj_epochs), 0)
            max_steps_per_epoch = int(self._proj_max_steps)
        else:
            epochs = max(int(self._proj_light_epochs), 0)
            max_steps_per_epoch = int(self._proj_light_max_steps)

        if max_steps_per_epoch <= 0:
            steps_per_epoch = int(len(train_loader))
        else:
            steps_per_epoch = min(int(len(train_loader)), max_steps_per_epoch)

        return {
            "enabled": True,
            "mode": mode,
            "epochs": epochs,
            "max_steps_per_epoch": max_steps_per_epoch,
            "estimated_extra_steps": int(epochs * steps_per_epoch),
        }

    # ------------------------------------------------------------------
    # Proof_Net-equivalent encode methods
    # ------------------------------------------------------------------

    def _encode_image_proj(self, raw_img_feat):
        """
        Mirrors Proof_Net.encode_image(normalize=False):
          stack all task heads, sum, return unnormalized.
        raw_img_feat: (B, d) normalized raw CLIP features.
        """
        feats = torch.stack([proj(raw_img_feat) for proj in self.projs_img], dim=1)  # (B, n_tasks, d)
        return feats.sum(dim=1)  # (B, d)

    def _encode_text_proj(self, raw_txt_feat):
        """Mirrors Proof_Net.encode_text(normalize=False)."""
        feats = torch.stack([proj(raw_txt_feat) for proj in self.projs_text], dim=1)  # (n_cls, n_tasks, d)
        return feats.sum(dim=1)  # (n_cls, d)

    def _encode_prototypes(self):
        """
        Mirrors Proof_Net.encode_prototpyes(normalize=True):
          stack all task heads, sum, normalize.
        """
        protos = self.img_prototypes  # (n_cls, d) — normalized raw CLIP features
        feats = torch.stack([proj(protos) for proj in self.projs_img], dim=1)  # (n_cls, n_tasks, d)
        return F.normalize(feats.sum(dim=1), dim=-1)  # (n_cls, d)

    def _forward_transformer(self, img_feat, txt_feat, proto_feat, context):
        """
        Mirrors Proof_Net.forward_transformer(transformer=True).
        img_feat:   (B, d)
        txt_feat:   (n_cls, d)
        proto_feat: (n_cls, d)
        context:    (n_ctx, d)
        Returns: (img_out (B,d), txt_out (n_cls,d), proto_out (n_cls,d))
        """
        B      = img_feat.shape[0]
        n_txt  = txt_feat.shape[0]
        n_proto = proto_feat.shape[0]

        img_exp   = img_feat.unsqueeze(1)                        # (B, 1, d)
        txt_exp   = txt_feat.unsqueeze(0).expand(B, -1, -1)     # (B, n_cls, d)
        proto_exp = proto_feat.unsqueeze(0).expand(B, -1, -1)   # (B, n_cls, d)
        ctx_exp   = context.unsqueeze(0).expand(B, -1, -1)      # (B, n_ctx, d)

        features = torch.cat([img_exp, txt_exp, proto_exp, ctx_exp], dim=1)
        features = self.sel_attn(features, features, features)

        img_out   = features[:, 0, :]                            # (B, d)
        txt_out   = features[:, 1:1+n_txt, :].mean(dim=0)       # (n_cls, d)
        proto_out = features[:, 1+n_txt:1+n_txt+n_proto, :].mean(dim=0)  # (n_cls, d)
        return img_out, txt_out, proto_out

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _cal_prototype(self, train_loader):
        """
        Mirrors proof.py::cal_prototype.
        Stores mean of *normalized* raw CLIP features per new class.
        (encode_image(data, True) in original = raw CLIP + L2 normalize)
        """
        self.clip_model.eval()
        embeddings, labels = [], []
        with torch.no_grad():
            for batch in train_loader:
                inputs, targets, _ = batch
                inputs = inputs.to(self.device)
                feat = self.clip_model.encode_image(inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)  # normalize=True
                embeddings.append(feat.cpu())
                labels.append(targets.cpu())
        embeddings = torch.cat(embeddings, dim=0)
        labels     = torch.cat(labels, dim=0)

        for cls_idx in range(self._known_classes, self._total_classes):
            mask = labels == cls_idx
            if mask.sum() > 0:
                self.img_prototypes[cls_idx] = embeddings[mask].mean(0).to(self.device)
        logging.info(f"[PROOF] prototypes updated for class range [{self._known_classes}, {self._total_classes})")

    @staticmethod
    def _clip_contrastive_loss(img_feat, txt_feat, logit_scale):
        """Symmetric contrastive loss — single-GPU ClipLoss from toolkit.py."""
        logits_i2t = logit_scale * img_feat @ txt_feat.t()
        logits_t2i = logits_i2t.t()
        labels = torch.arange(img_feat.shape[0], device=img_feat.device)
        return (F.cross_entropy(logits_i2t, labels) + F.cross_entropy(logits_t2i, labels)) / 2

    def _train_proj(self, train_loader, epochs: int, max_steps_per_epoch: int, mode: str):
        """
        Mirrors proof.py::_train_proj exactly.

        Freezing (freeze_projection_weight_new):
          - All projs_img[i] frozen, except projs_img[-1] unfrozen
          - All projs_text[i] unfrozen
          - sel_attn unfrozen
          - logit_scale unfrozen (convnet params frozen except logit_scale)

        Loss = CE(logits, targets) + ClipLoss(img_feas, clip_txt_feas, logit_scale)
                + CE(image_features @ proto_feas.T, targets)

        where:
          image_features = encode_image_proj(raw_img) / norm  (projected+normalized)
          text_features  = encode_text_proj(raw_txt) / norm   (projected+normalized)
          img_feas       = image_features (already normalized above)
          logits         = image_features @ text_features.T   (no logit_scale, matching proof.py line 139)
          clip_txt_feas  = encode_text_proj(per_sample_raw_txt) / norm
        """
        # --- Freeze backbone except logit_scale ---
        for name, param in self.clip_model.named_parameters():
            param.requires_grad = (name == 'logit_scale')

        # --- freeze_projection_weight_new ---
        for i in range(len(self.projs_img)):
            for p in self.projs_img[i].parameters():
                p.requires_grad = False
            for p in self.projs_text[i].parameters():
                p.requires_grad = True
        for p in self.projs_img[-1].parameters():
            p.requires_grad = True
        for p in self.sel_attn.parameters():
            p.requires_grad = True
        # Unfreeze new context prompt
        self.context_prompts[-1].requires_grad = True

        trainable = filter(lambda p: p.requires_grad, self.parameters())
        if self._proj_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                trainable,
                lr=self._proj_lr,
                momentum=self._proj_momentum,
                weight_decay=self._proj_weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                trainable,
                lr=self._proj_lr,
                weight_decay=self._proj_weight_decay,
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(epochs), 1), eta_min=self._min_lr
        )

        trainable_params_list = [p for p in self.parameters() if p.requires_grad]
        trainable_tensors = len(trainable_params_list)
        trainable_numel = int(sum(int(p.numel()) for p in trainable_params_list))

        all_text_tokens = self.text_tokens  # (n_cls, 77), all seen classes

        self.train()
        logging.info(
            f"[PROOF] projection training start: mode={mode}, epochs={epochs}, optimizer={self._proj_optimizer}, "
            f"max_steps_per_epoch={max_steps_per_epoch}, trainable_tensors={trainable_tensors}, trainable_params={trainable_numel}"
        )
        extra_steps = 0
        for epoch in range(int(epochs)):
            epoch_steps = 0
            for batch_idx, batch in enumerate(train_loader):
                inputs, targets, _ = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Raw normalized CLIP features (backbone frozen)
                with torch.no_grad():
                    raw_img = self.clip_model.encode_image(inputs)
                    raw_txt = self.clip_model.encode_text(all_text_tokens)

                # Projected features (mirrors encode_image / encode_text)
                img_proj = self._encode_image_proj(raw_img)   # (B, d) unnormalized
                txt_proj = self._encode_text_proj(raw_txt)    # (n_cls, d) unnormalized

                # Normalize (proof.py lines 135-136)
                img_feas = img_proj / img_proj.norm(dim=-1, keepdim=True)  # (B, d)
                txt_feas = txt_proj / txt_proj.norm(dim=-1, keepdim=True)  # (n_cls, d)

                proto_feat = self._encode_prototypes()  # (n_cls, d) normalized
                context    = torch.cat(list(self.context_prompts), dim=0)

                img_fused, txt_fused, proto_fused = self._forward_transformer(
                    img_feas, txt_feas, proto_feat, context
                )

                logit_scale = self.clip_model.logit_scale.exp()
                # proof.py line 139: logits = image_features @ text_features.T (no logit_scale)
                logits = img_fused @ txt_fused.t()

                # Per-sample text features for CLIP contrastive loss
                batch_names = [self.current_class_names[t] for t in targets.cpu().tolist()]
                batch_tokens = self.tokenizer(
                    [self._train_prompt_template.format(c) for c in batch_names]
                ).to(self.device)
                with torch.no_grad():
                    raw_clip_txt = self.clip_model.encode_text(batch_tokens)

                # clip_loss uses projected+normalized img_feas (proof.py line 146)
                clip_txt_proj = self._encode_text_proj(raw_clip_txt)
                clip_txt_feas = clip_txt_proj / clip_txt_proj.norm(dim=-1, keepdim=True)

                ce_loss = F.cross_entropy(logits, targets)
                proto_loss = F.cross_entropy(img_fused @ proto_fused.t(), targets)
                proj_ce_loss = F.cross_entropy(img_feas @ txt_feas.t(), targets)
                clip_loss = self._clip_contrastive_loss(img_feas, clip_txt_feas, logit_scale)

                if self._objective_mode == "paper_eq10":
                    total_loss = proj_ce_loss + ce_loss + proto_loss
                else:
                    total_loss = ce_loss + clip_loss + proto_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                extra_steps += 1
                epoch_steps += 1

                if batch_idx % self._proof_log_batch_interval == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    logging.info(
                        f"[PROOF][Task {self.current_task}] "
                        f"Epoch [{epoch + 1}/{epochs}] "
                        f"Batch [{batch_idx + 1}/{len(train_loader)}] | "
                        f"CE: {ce_loss.item():.4f} | "
                        f"ProjCE: {proj_ce_loss.item():.4f} | "
                        f"Clip: {clip_loss.item():.4f} | "
                        f"Proto: {proto_loss.item():.4f} | "
                        f"Total: {total_loss.item():.4f} | "
                        f"LR: {lr:.6e}"
                    )

                if int(max_steps_per_epoch) > 0 and epoch_steps >= int(max_steps_per_epoch):
                    break

            scheduler.step()

        # Re-freeze everything after training
        for p in self.clip_model.parameters():
            p.requires_grad = False
        for mod in list(self.projs_img) + list(self.projs_text):
            for p in mod.parameters():
                p.requires_grad = False
        for p in self.sel_attn.parameters():
            p.requires_grad = False
        for p in self.context_prompts:
            p.requires_grad = False
        logging.info(f"[PROOF] projection training complete and parameters refrozen | extra_steps={extra_steps}")
        return {
            "actual_extra_steps": int(extra_steps),
            "trainable_tensors": int(trainable_tensors),
            "trainable_params": int(trainable_numel),
        }
