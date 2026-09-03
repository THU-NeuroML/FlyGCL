import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Sampler

import clip
from continual_clip import utils
from continual_clip.OnlineIterDataset import OnlineIterDataset
from continual_clip.datasets import build_cl_scenarios, get_dataset_for_gcl
from continual_clip.gcl_metrics import GCLMetrics
from continual_clip.models import load_model
from continual_clip.utils.onlinesampler import OnlineSampler, OnlineTestSampler
from continual_clip.analysis import SeqLoRAAnalyzer
from continual_clip.analysis.schema_metadata import analysis_stats_run_metadata


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def _is_smoke_mode() -> bool:
    smoke_flag = os.environ.get("MULTI_SEED_SMOKE_MODE", "0").strip().lower()
    return smoke_flag in {"1", "true", "yes", "on"}


def _resolve_smoke_max_train_batches(default_batches: int = 1) -> int:
    if not _is_smoke_mode():
        return 0

    raw_value = os.environ.get("MULTI_SEED_SMOKE_MAX_BATCHES", str(default_batches)).strip()
    try:
        resolved = int(raw_value)
    except ValueError:
        resolved = default_batches
    return max(1, resolved)


def _resolve_smoke_max_eval_batches(default_batches: int = 4) -> int:
    if not _is_smoke_mode():
        return 0

    raw_value = os.environ.get("MULTI_SEED_SMOKE_MAX_EVAL_BATCHES", str(default_batches)).strip()
    try:
        resolved = int(raw_value)
    except ValueError:
        resolved = default_batches
    return max(1, resolved)


def _resolve_smoke_skip_after_task(default_skip: bool = True) -> bool:
    if not _is_smoke_mode():
        return False

    default_raw = "1" if default_skip else "0"
    raw = os.environ.get("MULTI_SEED_SMOKE_SKIP_AFTER_TASK", default_raw).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_cfg_paths(cfg: DictConfig) -> None:
    if "path" in cfg:
        for key, value in cfg.path.items():
            if key not in cfg or cfg[key] in ("", None):
                cfg[key] = value


def _ensure_seq_lora_analysis_defaults(cfg: DictConfig) -> None:
    defaults = {
        "enable_seq_lora_analysis": False,
        "analysis_interval": 1,
        "analysis_max_batches": 4,
        "analysis_output_dir_name": "analysis_seq_lora",
        "analysis_record_attention": True,
        "analysis_record_lora": True,
        "analysis_record_clip_alignment": True,
        "analysis_record_feature_drift": True,
        "analysis_record_grad_conflict": False,
        "analysis_record_layer_rollback": False,
        "enable_groupwise_analysis": False,
        "clip_ft_trainable_scope": "full",
        "prompt_trainable_scope": "all",
    }
    for key, value in defaults.items():
        if not hasattr(cfg, key):
            setattr(cfg, key, value)


def _is_seq_lora_method(method_name: str) -> bool:
    return str(method_name).lower() in {"lora", "seq_lora", "seq-lora"}


def _supports_diagnostic_analysis(method_name: str) -> bool:
    return str(method_name).lower() in {
        "lora",
        "seq_lora",
        "seq-lora",
        "l2p",
        "l2p_official",
        "dualprompt",
        "dual_prompt",
        "misa",
        "misa_l2p",
        "coda",
        "codaprompt",
        "coda_prompt",
        "clap4clip",
        "clap",
        "mindthegap",
        "mind_the_gap",
        "mtg",
        "lwf",
        "online_lwf",
        "ewc",
        "online_ewc",
        "ewc_kv",
        "clip_ft",
        "full_ft",
        "finetune",
        "fly_clip",
        "fly_clip_text_feature_ema",
        "fly_clip_text_linear_ema",
        "fly_clip_text_expert_head",
        "fly_clip_text_expert_ema",
        "fly_clip_text_feature_linear_ema",
    }


def _clip_contrastive_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute CLIP-style symmetric contrastive loss from class logits.

    logits: [B, C] image-to-class similarities.
    labels: [B] class indices in [0, C).
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits [B,C], got shape={tuple(logits.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels [B], got shape={tuple(labels.shape)}")
    if logits.size(0) == 0:
        return logits.sum() * 0.0

    labels = labels.long()
    pair_logits = logits[:, labels]  # [B, B], column j is text of sample j's class
    targets = torch.arange(pair_logits.size(0), device=pair_logits.device)
    loss_i2t = F.cross_entropy(pair_logits, targets)
    loss_t2i = F.cross_entropy(pair_logits.t(), targets)
    return 0.5 * (loss_i2t + loss_t2i)


def _normalize_prompt_primary_loss(raw_loss: str) -> str:
    mode = str(raw_loss).lower()
    if mode in {"clip_pair", "clip", "contrastive", "clip_contrastive"}:
        return "clip_pair"
    if mode in {"ce", "cross_entropy", "class_ce"}:
        return "ce"
    logging.info(
        f"[PROMPT LOSS] unknown prompt_primary_loss={raw_loss}; fallback to clip_pair"
    )
    return "clip_pair"


def _primary_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, loss_type: str) -> torch.Tensor:
    mode = _normalize_prompt_primary_loss(loss_type)
    if mode == "ce":
        return F.cross_entropy(logits, labels)
    return _clip_contrastive_loss_from_logits(logits, labels)


def _normalize_lwf_primary_loss(raw_loss: str) -> str:
    mode = str(raw_loss).lower()
    if mode in {"ce", "cross_entropy", "class_ce"}:
        return "ce"
    if mode in {"clip_pair", "clip", "contrastive", "clip_contrastive"}:
        return "clip_pair"
    logging.info(f"[LWF LOSS] unknown lwf_primary_loss={raw_loss}; fallback to ce")
    return "ce"


def _trainable_param_summary(model) -> Dict[str, int]:
    summary = {
        "total_params": 0,
        "trainable_params": 0,
        "visual_trainable_params": 0,
        "text_trainable_params": 0,
        "logit_scale_trainable_params": 0,
    }
    for name, param in model.named_parameters():
        n = int(param.numel())
        summary["total_params"] += n
        if not param.requires_grad:
            continue
        summary["trainable_params"] += n
        if "logit_scale" in name:
            summary["logit_scale_trainable_params"] += n
        elif name.startswith("visual.") or name.startswith("model.visual."):
            summary["visual_trainable_params"] += n
        else:
            summary["text_trainable_params"] += n
    return summary


def _method_aux_string(aux_info: Dict, cfg_method: str) -> str:
    if not aux_info:
        return ""

    method_name = str(aux_info.get("method", cfg_method)).lower()
    if method_name in {"l2p", "l2p_official", "dualprompt", "dualprompt_official", "misa", "misa_l2p"}:
        reduce_sim = aux_info.get("reduce_sim", None)
        pull_enabled = aux_info.get("pull_constraint", False)
        pull_coeff = aux_info.get("pull_constraint_coeff", None)
        prompt_window = (
            aux_info.get("prompt_window_start", None),
            aux_info.get("prompt_window_end", None),
        )
        prompt_window_str = (
            f"[{prompt_window[0]}, {prompt_window[1]})"
            if (prompt_window[0] is not None and prompt_window[1] is not None)
            else "NA"
        )
        return (
            f" | Pull: {int(bool(pull_enabled))}"
            f" | PullCoeff: {pull_coeff if pull_coeff is not None else 'NA'}"
            f" | ReduceSim: {reduce_sim if reduce_sim is not None else 'NA'}"
            f" | PromptWindow: {prompt_window_str}"
        )

    if method_name == "coda":
        ortho_mu = aux_info.get("ortho_mu", None)
        ortho_active = aux_info.get("ortho_penalty_active", None)
        prompt_loss = aux_info.get("prompt_loss", None)
        return (
            f" | PromptLoss: {prompt_loss if prompt_loss is not None else 'NA'}"
            f" | OrthoMu: {ortho_mu if ortho_mu is not None else 'NA'}"
            f" | OrthoPenaltyActive: {int(bool(ortho_active)) if ortho_active is not None else 'NA'}"
        )

    if method_name == "lwf":
        kd = aux_info.get("kd", None)
        distill_lambda = aux_info.get("distill_lambda", None)
        distill_temp = aux_info.get("distill_temp", None)
        teacher_active = aux_info.get("teacher_active", None)
        return (
            f" | KD: {kd if kd is not None else 'NA'}"
            f" | DistillLambda: {distill_lambda if distill_lambda is not None else 'NA'}"
            f" | DistillTemp: {distill_temp if distill_temp is not None else 'NA'}"
            f" | TeacherActive: {int(bool(teacher_active)) if teacher_active is not None else 'NA'}"
        )

    if method_name in {"ewc", "online_ewc", "ewc_kv"}:
        ewc = aux_info.get("ewc", None)
        ewc_lambda = aux_info.get("ewc_lambda", None)
        fisher_active = aux_info.get("fisher_active", None)
        retention_mode = aux_info.get("ewc_retention_mode", None)
        retained_anchors = aux_info.get("retained_anchors", None)
        active_anchors = aux_info.get("active_anchors", None)
        return (
            f" | EWC: {ewc if ewc is not None else 'NA'}"
            f" | EwcLambda: {ewc_lambda if ewc_lambda is not None else 'NA'}"
            f" | FisherActive: {int(bool(fisher_active)) if fisher_active is not None else 'NA'}"
            f" | EwcRetention: {retention_mode if retention_mode is not None else 'NA'}"
            f" | RetainedAnchors: {retained_anchors if retained_anchors is not None else 'NA'}"
            f" | ActiveAnchors: {active_anchors if active_anchors is not None else 'NA'}"
        )

    if method_name == "fly":
        fly_mode = aux_info.get("fly_mode", None)
        route_entropy = aux_info.get("route_entropy", None)
        routed_top = aux_info.get("routed_expert_top", None)
        ema_updates = aux_info.get("ema_updates", None)
        rp_samples = aux_info.get("rp_samples_accum", None)
        return (
            f" | FlyMode: {fly_mode if fly_mode is not None else 'NA'}"
            f" | RouteEntropy: {route_entropy if route_entropy is not None else 'NA'}"
            f" | RoutedTop: {routed_top if routed_top is not None else 'NA'}"
            f" | EmaUpdates: {ema_updates if ema_updates is not None else 'NA'}"
            f" | RpSamples: {rp_samples if rp_samples is not None else 'NA'}"
        )

    return ""


def _build_session_class_plan(sampler: OnlineSampler, num_sessions: int) -> List[List[int]]:
    plan = []
    for session_id in range(num_sessions):
        disjoint = sampler.disjoint_classes[session_id] if session_id < len(sampler.disjoint_classes) else []
        blurry = sampler.blurry_classes[session_id] if session_id < len(sampler.blurry_classes) else []
        merged = sorted(set(int(x) for x in list(disjoint) + list(blurry)))
        plan.append(merged)
    return plan


def _seen_class_ids_from_model(model, class_to_idx: Dict[str, int], session_plan: Sequence[Sequence[int]], session_id: int) -> List[int]:
    if hasattr(model, "current_class_ids") and model.current_class_ids is not None:
        ids = [int(x) for x in list(model.current_class_ids)]
        if ids:
            # Keep insertion order while de-duplicating.
            return list(dict.fromkeys(ids))

    if hasattr(model, "current_class_names") and model.current_class_names is not None:
        ids = []
        missing = []
        for name in model.current_class_names:
            if name in class_to_idx:
                ids.append(class_to_idx[name])
            else:
                missing.append(name)
        if ids and not missing:
            return list(dict.fromkeys(int(x) for x in ids))
        if missing:
            logging.warning(
                "[PROOF] current_class_names has unmapped labels; fallback to session plan. "
                f"missing={len(missing)} sample={missing[:5]}"
            )

    seen = set()
    for i in range(session_id + 1):
        seen.update(session_plan[i])
    return sorted(seen)


def _align_exposed_with_logits(exposed: List[int], logits_dim: int) -> List[int]:
    if logits_dim <= 0:
        return []
    if len(exposed) == logits_dim:
        return exposed
    if len(exposed) > logits_dim:
        return exposed[:logits_dim]
    return exposed


def _session_old_new_exposed_acc(cls_acc: np.ndarray, session_plan: Sequence[Sequence[int]], session_id: int):
    old_ids = set()
    for i in range(session_id):
        old_ids.update(int(x) for x in session_plan[i])
    new_ids = set(int(x) for x in session_plan[session_id]) if session_id < len(session_plan) else set()
    exposed_ids = old_ids | new_ids

    def _mean_acc(ids_set):
        ids = sorted(list(ids_set))
        if len(ids) == 0:
            return None
        values = [float(cls_acc[i]) for i in ids]
        return float(np.mean(values))

    return {
        "old_exposed_acc": _mean_acc(old_ids),
        "new_exposed_acc": _mean_acc(new_ids),
        "all_exposed_acc": _mean_acc(exposed_ids),
    }


def _is_prompt_family(method_name: str) -> bool:
    return method_name.lower() in {"l2p", "l2p_official", "dualprompt", "dualprompt_official", "misa", "misa_l2p", "coda"}


def _resolve_aux_loss_coeff(cfg: DictConfig, method_name: str) -> float:
    if _is_prompt_family(method_name):
        return float(getattr(cfg, "prompt_aux_loss_coeff", 1.0))
    return float(getattr(cfg, "aux_loss_coeff", 1.0))


def _normalize_method_protocol_mode(raw_mode: str, method_name: str, default_mode: str = "strict_session_task") -> str:
    mode = str(raw_mode).lower()
    if mode in {"strict", "strict_session_task"}:
        return "strict_session_task"
    if mode in {"faithful", "faithful_session_task"}:
        return "faithful_session_task"
    logging.info(
        f"[PROTOCOL] unknown {method_name}_protocol_mode={raw_mode}; fallback to {default_mode}"
    )
    return default_mode


def _resolve_method_protocol_mode(cfg: DictConfig, method_name: str, strict_budget_mode: bool) -> str:
    m = method_name.lower()
    if _is_prompt_family(m):
        return str(getattr(cfg, "prompt_protocol_mode", "faithful_session_task")).lower()
    if m == "proof":
        return str(getattr(cfg, "proof_protocol_mode", "strict_session_task")).lower()
    if m in {"ewc", "online_ewc", "ewc_kv"}:
        return _normalize_method_protocol_mode(
            raw_mode=getattr(cfg, "ewc_protocol_mode", "strict"),
            method_name="ewc",
            default_mode="strict_session_task",
        )
    if m == "lwf":
        return _normalize_method_protocol_mode(
            raw_mode=getattr(cfg, "lwf_protocol_mode", "strict"),
            method_name="lwf",
            default_mode="strict_session_task",
        )
    return "strict_session_task"


def _resolve_prompt_profile(cfg: DictConfig) -> str:
    profile = str(getattr(cfg, "prompt_profile", "gcl_optimized")).lower()
    if profile not in {"paper_faithful", "gcl_optimized"}:
        logging.info(
            f"[PROMPT PROFILE] unknown prompt_profile={profile}; fallback to gcl_optimized"
        )
        profile = "gcl_optimized"
    cfg.prompt_profile = profile
    return profile


def _prompt_profile_overrides(method_name: str, profile: str) -> Dict[str, object]:
    method = method_name.lower()

    # Source provenance for CODA ortho coefficient:
    # - external/CODA-Prompt-main/run.py parser default: [1, 1, 1]
    # - external experiment scripts use 0.0 under the updated Gram-Schmidt variant.
    # We keep 1.0 as the strict non-zero "paper_faithful" fallback and 0.0 for optimized.
    coda_faithful_ortho = 1.0

    if method in {"l2p", "l2p_official", "misa_l2p"}:
        if profile == "paper_faithful":
            return {
                "prompt_top_k": 4,
                "prompt_mask": True,
                "pull_constraint": True,
                "pull_constraint_coeff": 1.0,
                "prompt_window_mode": "hard_session",
                "prompt_eval_mode": "same_as_train",
                "prompt_mask_old_logits": True,
                "prompt_train_on_old_classes": False,
            }
        return {
            "prompt_top_k": 4,
            "prompt_mask": True,
            "pull_constraint": True,
            "pull_constraint_coeff": 0.5,
            "prompt_window_mode": "global",
            "prompt_eval_mode": "global",
            "prompt_mask_old_logits": False,
            "prompt_train_on_old_classes": True,
        }

    if method in {"dualprompt", "dualprompt_official", "misa"}:
        if profile == "paper_faithful":
            return {
                "prompt_top_k": 1,
                "use_g_prompt": True,
                "use_e_prompt": True,
                "prompt_mask": True,
                "pull_constraint": True,
                "pull_constraint_coeff": 1.0,
                "prompt_window_mode": "hard_session",
                "prompt_eval_mode": "same_as_train",
                "prompt_mask_old_logits": True,
                "prompt_train_on_old_classes": False,
            }
        return {
            "prompt_top_k": 4,
            "use_g_prompt": True,
            "use_e_prompt": True,
            "prompt_mask": True,
            "pull_constraint": True,
            "pull_constraint_coeff": 0.5,
            "prompt_window_mode": "global",
            "prompt_eval_mode": "global",
            "prompt_mask_old_logits": False,
            "prompt_train_on_old_classes": True,
        }

    if method == "coda":
        if profile == "paper_faithful":
            return {
                "prompt_param": [100, 8, coda_faithful_ortho],
                "prompt_window_mode": "hard_session",
                "prompt_eval_mode": "same_as_train",
                "prompt_mask_old_logits": True,
                "prompt_train_on_old_classes": False,
            }
        return {
            "prompt_param": [100, 8, 0.0],
            "prompt_window_mode": "global",
            "prompt_eval_mode": "global",
            "prompt_mask_old_logits": False,
            "prompt_train_on_old_classes": True,
        }

    return {}


def _apply_prompt_overrides(cfg: DictConfig, overrides: Dict[str, object]) -> None:
    for key, value in overrides.items():
        setattr(cfg, key, value)


def _apply_prompt_protocol_policy(cfg: DictConfig) -> str:
    mode = str(getattr(cfg, "prompt_protocol_mode", "faithful_session_task")).lower()
    if mode not in {"strict_session_task", "faithful_session_task"}:
        logging.info(
            f"[PROMPT PROTOCOL] unknown prompt_protocol_mode={mode}; fallback to strict_session_task"
        )
        mode = "strict_session_task"
        cfg.prompt_protocol_mode = "strict_session_task"

    if mode == "strict_session_task":
        cfg.prompt_mask_old_logits = True
        cfg.prompt_train_on_old_classes = False
        cfg.prompt_window_mode = "hard_session"
        cfg.prompt_eval_mode = "same_as_train"
        logging.info(
            "[PROMPT PROTOCOL] strict_session_task enforced | "
            "prompt_train_on_old_classes=0 prompt_mask_old_logits=1 "
            "prompt_window_mode=hard_session prompt_eval_mode=same_as_train"
        )
    else:
        cfg.prompt_train_on_old_classes = True
        cfg.prompt_mask_old_logits = False
        cfg.prompt_window_mode = "global"
        cfg.prompt_eval_mode = "global"
        logging.info(
            "[PROMPT PROTOCOL] faithful_session_task enforced | "
            "prompt_train_on_old_classes=1 prompt_mask_old_logits=0 "
            "prompt_window_mode=global prompt_eval_mode=global"
        )
    return mode


def _apply_ewc_protocol_policy(cfg: DictConfig) -> Dict[str, object]:
    requested = str(getattr(cfg, "ewc_protocol_mode", "strict")).lower()
    resolved = _normalize_method_protocol_mode(
        raw_mode=requested,
        method_name="ewc",
        default_mode="strict_session_task",
    )

    requested_retention = str(getattr(cfg, "ewc_retention_mode", "online")).lower()
    if requested_retention not in {"online", "taskwise"}:
        logging.info(
            f"[EWC PROTOCOL] unknown ewc_retention_mode={requested_retention}; fallback to online"
        )
        requested_retention = "online"

    retention_effective = requested_retention
    policy_overridden = False
    policy_reason = "user_authoritative"

    if resolved == "strict_session_task":
        retention_target = "online"
    else:
        retention_target = "taskwise"

    if requested_retention != retention_target:
        cfg.ewc_retention_mode = retention_target
        retention_effective = retention_target
        policy_overridden = True
        if resolved == "strict_session_task":
            policy_reason = "strict_forces_online_retention"
        else:
            policy_reason = "faithful_forces_taskwise_retention"
    else:
        cfg.ewc_retention_mode = retention_target

    cfg.ewc_protocol_mode = resolved
    logging.info(
        "[EWC POLICY] authoritative=ewc_protocol_mode "
        f"requested={requested} effective={str(cfg.ewc_protocol_mode)} "
        f"retention_requested={requested_retention} retention_effective={retention_effective} "
        f"overridden={int(bool(policy_overridden))} reason={policy_reason}"
    )
    return {
        "ewc_protocol_requested": str(requested),
        "ewc_protocol_effective": str(cfg.ewc_protocol_mode),
        "ewc_retention_requested": str(requested_retention),
        "ewc_retention_effective": str(retention_effective),
        "ewc_protocol_policy_overridden": int(bool(policy_overridden)),
        "ewc_protocol_policy_reason": str(policy_reason),
    }


def _resolved_prompt_audit(cfg: DictConfig, method_name: str) -> Dict[str, object]:
    base = {
        "method": method_name,
        "prompt_profile": str(getattr(cfg, "prompt_profile", "gcl_optimized")),
        "prompt_apply_profile_overrides": bool(getattr(cfg, "prompt_apply_profile_overrides", True)),
        "prompt_protocol_mode": str(getattr(cfg, "prompt_protocol_mode", "faithful_session_task")),
        "prompt_window_mode": str(getattr(cfg, "prompt_window_mode", "NA")),
        "prompt_eval_mode": str(getattr(cfg, "prompt_eval_mode", "NA")),
        "prompt_mask_old_logits": bool(getattr(cfg, "prompt_mask_old_logits", False)),
        "prompt_train_on_old_classes": bool(getattr(cfg, "prompt_train_on_old_classes", True)),
        "prompt_primary_loss": str(getattr(cfg, "prompt_primary_loss", "clip_pair")),
        "prompt_modalities": str(getattr(cfg, "prompt_modalities", getattr(cfg, "prompt_modality", "vision"))),
        "prompt_inject_all_layers": bool(getattr(cfg, "prompt_inject_all_layers", True)),
        "prompt_inject": str(getattr(cfg, "prompt_inject", "attention_kv_prefix")),
        "vision_prompt_inject": str(getattr(cfg, "vision_prompt_inject", getattr(cfg, "prompt_inject", "attention_kv_prefix"))),
        "text_prompt_inject": str(getattr(cfg, "text_prompt_inject", getattr(cfg, "prompt_inject", "attention_kv_prefix"))),
        "prompt_trainable_scope": str(getattr(cfg, "prompt_trainable_scope", "all")),
    }
    m = method_name.lower()
    if m in {"l2p", "l2p_official", "dualprompt", "dualprompt_official", "misa", "misa_l2p"}:
        base["prompt_top_k"] = int(getattr(cfg, "prompt_top_k", 0))
        base["prompt_mask"] = bool(getattr(cfg, "prompt_mask", False))
        base["pull_constraint"] = bool(getattr(cfg, "pull_constraint", False))
        base["pull_constraint_coeff"] = float(getattr(cfg, "pull_constraint_coeff", 0.0))
    if m in {"dualprompt", "dualprompt_official", "misa"}:
        base["use_g_prompt"] = bool(getattr(cfg, "use_g_prompt", True))
        base["use_e_prompt"] = bool(getattr(cfg, "use_e_prompt", True))
    if m in {"misa", "misa_l2p"}:
        base["misa_logit_mask"] = bool(getattr(cfg, "misa_logit_mask", True))
    if m == "coda":
        base["prompt_param"] = [float(x) for x in list(getattr(cfg, "prompt_param", [100, 8, 0.0]))]
        base["coda_ortho_mu"] = float(base["prompt_param"][2]) if len(base["prompt_param"]) >= 3 else 0.0
    return base


def _validate_prompt_hard_session_capacity(cfg: DictConfig, method_name: str) -> None:
    method = str(method_name).lower()
    if method not in {"l2p", "l2p_official", "dualprompt", "dualprompt_official", "misa", "misa_l2p"}:
        return

    window_mode = str(getattr(cfg, "prompt_window_mode", "global")).lower()
    use_prompt_mask = bool(getattr(cfg, "prompt_mask", False))
    if window_mode != "hard_session" or not use_prompt_mask:
        return

    task_num = int(getattr(cfg, "task_num", getattr(cfg, "gcl_sessions", 1)))
    top_k = int(getattr(cfg, "prompt_top_k", 1))
    if method in {"l2p", "l2p_official", "misa_l2p"}:
        pool_size = int(getattr(cfg, "prompt_pool_size", 0))
    else:
        pool_size = int(getattr(cfg, "e_pool_size", 0))

    required = int(task_num) * int(top_k)
    if pool_size < required:
        raise ValueError(
            "[PROMPT WINDOW] hard_session requires sufficient prompt capacity: "
            f"method={method} pool_size={pool_size} required={required} "
            f"(task_num={task_num} * prompt_top_k={top_k})"
        )


def _protocol_bucket(protocol_mode: str) -> str:
    mode = str(protocol_mode).lower()
    if mode == "paper_full" or mode.startswith("paper"):
        return "paper_full"
    if mode.startswith("faithful"):
        return "faithful"
    if mode.startswith("strict"):
        return "strict"
    return "strict"


def _compute_session_transfer_metrics(
    cls_accs: np.ndarray,
    session_plan: Sequence[Sequence[int]],
    class_intro_session: Dict[int, int],
    session_id: int,
) -> Tuple[float, float, float]:
    old_ids = set()
    for i in range(session_id):
        old_ids.update(int(x) for x in session_plan[i])
    new_ids = set(int(x) for x in session_plan[session_id]) if session_id < len(session_plan) else set()

    forgetting_vals = []
    bwt_vals = []
    for cls_id in sorted(old_ids):
        prev = cls_accs[:session_id, cls_id]
        if prev.size == 0:
            continue
        peak_before = float(np.max(prev))
        current = float(cls_accs[session_id, cls_id])
        forgetting_vals.append(peak_before - current)

        intro_s = int(class_intro_session.get(cls_id, 0))
        intro_s = max(0, min(intro_s, session_id))
        first_acc = float(cls_accs[intro_s, cls_id])
        bwt_vals.append(current - first_acc)

    if session_id > 0 and len(new_ids) > 0:
        fwt_vals = [float(cls_accs[session_id - 1, cls_id]) for cls_id in sorted(new_ids)]
        fwt = float(np.mean(fwt_vals)) if len(fwt_vals) > 0 else 0.0
    else:
        fwt = 0.0

    forgetting = float(np.mean(forgetting_vals)) if len(forgetting_vals) > 0 else 0.0
    bwt = float(np.mean(bwt_vals)) if len(bwt_vals) > 0 else 0.0
    return forgetting, bwt, fwt


def _build_optimizer(trainable_params, cfg: DictConfig):
    optimizer_name = str(getattr(cfg, "optimizer", "adamw")).lower()
    lr = float(cfg.lr)
    weight_decay = float(getattr(cfg, "weight_decay", 0.0))
    momentum = float(getattr(cfg, "momentum", 0.9))
    beta1 = float(getattr(cfg, "beta1", 0.9))
    beta2 = float(getattr(cfg, "beta2", 0.999))
    eps = float(getattr(cfg, "optimizer_eps", 1e-8))
    nesterov = bool(getattr(cfg, "nesterov", False))

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_params,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Choose from ['sgd', 'adam', 'adamw']")


def _build_scheduler(optimizer, total_steps: int, cfg: DictConfig):
    scheduler_name = str(getattr(cfg, "scheduler", "cosine")).lower()
    if scheduler_name in {"none", "constant"}:
        return None, 0

    warmup_steps_cfg = int(getattr(cfg, "warmup_steps", -1))
    if warmup_steps_cfg >= 0:
        warmup_steps = min(max(warmup_steps_cfg, 0), max(total_steps - 1, 0))
    else:
        warmup_ratio = float(getattr(cfg, "warmup_ratio", 0.0))
        warmup_steps = min(int(total_steps * warmup_ratio), max(total_steps - 1, 0))

    min_lr_ratio = float(getattr(cfg, "min_lr_ratio", 0.0))
    min_lr_ratio = min(max(min_lr_ratio, 0.0), 1.0)
    denom = max(total_steps - warmup_steps, 1)

    if scheduler_name == "cosine":
        def _lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(denom)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        return scheduler, warmup_steps

    if scheduler_name == "linear":
        def _lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(denom)
            progress = min(max(progress, 0.0), 1.0)
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        return scheduler, warmup_steps

    raise ValueError(f"Unsupported scheduler '{scheduler_name}'. Choose from ['none', 'constant', 'linear', 'cosine']")


class _FixedIndexSampler(Sampler):
    def __init__(self, indices: Sequence[int]):
        self.indices = [int(i) for i in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class _ProofIndexReplayMemory:
    """Class-balanced replay memory storing dataset indices for GCL PROOF paper_full mode."""

    def __init__(self, memory_size: int, memory_per_class: int, fixed_memory: bool, seed: int = 0):
        self.memory_size = max(int(memory_size), 0)
        self.memory_per_class = max(int(memory_per_class), 1)
        self.fixed_memory = bool(fixed_memory)
        self.rng = random.Random(int(seed))
        self._class_to_indices = defaultdict(list)

    def _budget_per_class(self, num_seen_classes: int) -> int:
        if self.fixed_memory:
            return max(int(self.memory_per_class), 1)
        seen = max(int(num_seen_classes), 1)
        if self.memory_size <= 0:
            return 0
        return max(int(self.memory_size // seen), 1)

    def get_indices(self) -> List[int]:
        out = []
        for cls_id in sorted(self._class_to_indices.keys()):
            out.extend(int(i) for i in self._class_to_indices[cls_id])
        return out

    def update_with_session(
        self,
        session_indices: Sequence[int],
        targets: Sequence[int],
        new_class_ids: Sequence[int],
        total_seen_classes: int,
    ) -> None:
        if self.memory_size <= 0:
            self._class_to_indices = defaultdict(list)
            return

        new_class_set = set(int(c) for c in new_class_ids)
        grouped = defaultdict(list)
        for idx in session_indices:
            i = int(idx)
            cls = int(targets[i])
            if cls in new_class_set:
                grouped[cls].append(i)

        for cls, idxs in grouped.items():
            merged = list(self._class_to_indices[cls]) + list(idxs)
            dedup = list(dict.fromkeys(int(i) for i in merged))
            self.rng.shuffle(dedup)
            self._class_to_indices[cls] = dedup

        budget = self._budget_per_class(total_seen_classes)
        if budget <= 0:
            self._class_to_indices = defaultdict(list)
            return

        all_classes = list(self._class_to_indices.keys())
        for cls in all_classes:
            kept = list(self._class_to_indices[cls])[:budget]
            if len(kept) == 0:
                self._class_to_indices.pop(cls, None)
            else:
                self._class_to_indices[cls] = kept


class _ProofSampleReplayMemory:
    """Class-balanced sample replay memory mirroring source PROOF appendent behavior."""

    def __init__(self, memory_size: int, memory_per_class: int, fixed_memory: bool):
        self.memory_size = max(int(memory_size), 0)
        self.memory_per_class = max(int(memory_per_class), 1)
        self.fixed_memory = bool(fixed_memory)
        self.mem_x = np.array([])
        self.mem_y = np.array([])
        self.mem_t = np.array([])

    def get(self):
        if len(self.mem_y) == 0:
            return None, None, None
        return self.mem_x, self.mem_y, self.mem_t

    def add_task(self, raw_x, raw_y, raw_t, total_classes: int):
        if raw_x is None or raw_y is None:
            return

        raw_x = np.asarray(raw_x)
        raw_y = np.asarray(raw_y)
        if raw_t is None:
            raw_t = np.zeros_like(raw_y)
        else:
            raw_t = np.asarray(raw_t)

        if self.fixed_memory:
            per_class = int(self.memory_per_class)
        else:
            seen = max(int(total_classes), 1)
            if self.memory_size <= 0:
                per_class = 0
            else:
                per_class = max(int(self.memory_size // seen), 1)

        if per_class <= 0:
            self.mem_x = np.array([])
            self.mem_y = np.array([])
            self.mem_t = np.array([])
            return

        if len(self.mem_y) > 0:
            all_x = np.concatenate([self.mem_x, raw_x], axis=0)
            all_y = np.concatenate([self.mem_y, raw_y], axis=0)
            all_t = np.concatenate([self.mem_t, raw_t], axis=0)
        else:
            all_x, all_y, all_t = raw_x, raw_y, raw_t

        kept_idx = []
        for cls in np.unique(all_y):
            idx = np.where(all_y == cls)[0]
            if len(idx) > per_class:
                idx = np.random.choice(idx, size=per_class, replace=False)
            kept_idx.append(idx)

        if len(kept_idx) == 0:
            return

        kept_idx = np.concatenate(kept_idx, axis=0)
        self.mem_x = all_x[kept_idx]
        self.mem_y = all_y[kept_idx]
        self.mem_t = all_t[kept_idx]


def _resolve_source_increments(num_classes: int, cfg: DictConfig) -> Tuple[int, int, List[int]]:
    init_cls = int(getattr(cfg, "proof_source_initial_increment", -1))
    increment = int(getattr(cfg, "proof_source_increment", -1))

    if init_cls <= 0:
        cfg_init = getattr(cfg, "initial_increment", None)
        init_cls = int(cfg_init) if cfg_init not in (None, "") else -1
    if increment <= 0:
        cfg_inc = getattr(cfg, "increment", None)
        increment = int(cfg_inc) if cfg_inc not in (None, "") else -1

    if init_cls <= 0 or increment <= 0:
        sessions = max(int(getattr(cfg, "gcl_sessions", 1)), 1)
        base = max(int(num_classes // sessions), 1)
        init_cls = base
        increment = base

    increments = [int(init_cls)]
    while sum(increments) + int(increment) < int(num_classes):
        increments.append(int(increment))
    offset = int(num_classes) - int(sum(increments))
    if offset > 0:
        increments.append(int(offset))
    return int(init_cls), int(increment), increments


def _build_source_session_plan(class_order: Sequence[int], increments: Sequence[int]) -> List[List[int]]:
    plan = []
    offset = 0
    ordered = list(int(x) for x in class_order)
    for inc in increments:
        nxt = offset + int(inc)
        plan.append(ordered[offset:nxt])
        offset = nxt
    return plan


def _build_proof_after_task_loader(base_loader, model, class_to_idx: Dict[str, int]):
    """
    PROOF after_task expects local class ids [0..num_seen-1], while GCL stream yields global class ids.
    Wrap the session loader to remap labels into PROOF local index space without touching method internals.
    """
    ordered_global_ids = []

    class_ids = list(getattr(model, "current_class_ids", []) or [])
    if class_ids:
        ordered_global_ids = [int(x) for x in class_ids]
    else:
        class_names = list(getattr(model, "current_class_names", []) or [])
        if len(class_names) == 0:
            return base_loader
        missing = []
        for name in class_names:
            if name in class_to_idx:
                ordered_global_ids.append(int(class_to_idx[name]))
            else:
                missing.append(name)
        if missing:
            logging.warning(
                "[PROOF] after_task name-based remap has unmapped labels; drop missing names. "
                f"missing={len(missing)} sample={missing[:5]}"
            )

    if len(ordered_global_ids) == 0:
        return base_loader

    ordered_global_ids = list(dict.fromkeys(ordered_global_ids))

    global_to_local = {gid: idx for idx, gid in enumerate(ordered_global_ids)}

    class _ProofAfterTaskLoader:
        def __init__(self, loader, mapping):
            self.loader = loader
            self.mapping = mapping

        def __len__(self):
            return len(self.loader)

        def __iter__(self):
            for images, labels, indices in self.loader:
                local_labels_list = []
                keep_mask = []
                for label in labels.tolist():
                    gid = int(label)
                    if gid in self.mapping:
                        keep_mask.append(True)
                        local_labels_list.append(self.mapping[gid])
                    else:
                        keep_mask.append(False)

                if not any(keep_mask):
                    continue

                keep_mask_t = torch.tensor(keep_mask, dtype=torch.bool, device=labels.device)
                images_kept = images[keep_mask_t]
                indices_kept = indices[keep_mask_t]
                labels_local = torch.tensor(local_labels_list, dtype=labels.dtype, device=labels.device)
                yield images_kept, labels_local, indices_kept

    return _ProofAfterTaskLoader(base_loader, global_to_local)


def evaluate_gcl(model, test_loader, exposed_class_ids, device) -> float:
    model.eval()
    correct = 0
    total = 0
    smoke_max_eval_batches = _resolve_smoke_max_eval_batches()
    eval_batches = 0

    with torch.no_grad():
        for images, labels, _ in test_loader:
            eval_batches += 1
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images, test=True, all_test=False)
            exposed = _align_exposed_with_logits(list(exposed_class_ids), int(logits.shape[1]))
            if not exposed:
                continue

            effective_dim = min(int(logits.shape[1]), len(exposed))
            if effective_dim <= 0:
                continue
            logits = logits[:, :effective_dim]
            exposed = exposed[:effective_dim]

            preds_local = logits.argmax(dim=1)
            preds_global = torch.tensor([exposed[idx] for idx in preds_local.cpu().tolist()], device=device)
            valid_mask = torch.tensor([int(lbl.item()) in set(exposed) for lbl in labels], dtype=torch.bool, device=device)
            if torch.any(valid_mask):
                correct += (preds_global[valid_mask] == labels[valid_mask]).sum().item()
                total += int(valid_mask.sum().item())

            if smoke_max_eval_batches > 0 and eval_batches >= smoke_max_eval_batches:
                break

    model.train()
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate_gcl_detailed(model, test_loader, exposed_class_ids, num_classes, device) -> Tuple[float, np.ndarray]:
    model.eval()
    cls_correct = np.zeros(num_classes, dtype=np.float64)
    cls_total = np.zeros(num_classes, dtype=np.float64)
    smoke_max_eval_batches = _resolve_smoke_max_eval_batches()
    eval_batches = 0

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in test_loader:
            eval_batches += 1
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images, test=True, all_test=False)

            exposed = _align_exposed_with_logits(list(exposed_class_ids), int(logits.shape[1]))
            if not exposed:
                continue

            effective_dim = min(int(logits.shape[1]), len(exposed))
            if effective_dim <= 0:
                continue
            logits = logits[:, :effective_dim]
            exposed = exposed[:effective_dim]

            exposed_set = set(exposed)
            preds_local = logits.argmax(dim=1)
            preds_global = torch.tensor([exposed[idx] for idx in preds_local.cpu().tolist()], device=device)

            valid_mask = torch.tensor([int(lbl.item()) in exposed_set for lbl in labels], dtype=torch.bool, device=device)
            if not torch.any(valid_mask):
                continue

            labels_valid = labels[valid_mask]
            preds_valid = preds_global[valid_mask]

            correct += int((preds_valid == labels_valid).sum().item())
            total += int(labels_valid.shape[0])

            for i in range(int(labels_valid.shape[0])):
                cls_id = int(labels_valid[i].item())
                cls_total[cls_id] += 1
                if int(preds_valid[i].item()) == cls_id:
                    cls_correct[cls_id] += 1

            if smoke_max_eval_batches > 0 and eval_batches >= smoke_max_eval_batches:
                break

    overall_acc = 100.0 * correct / total if total > 0 else 0.0
    cls_acc = np.divide(cls_correct, cls_total, out=np.zeros_like(cls_correct), where=cls_total > 0) * 100.0
    model.train()
    return overall_acc, cls_acc


def _build_oracle_expert_ids_from_labels(
    labels: torch.Tensor,
    class_to_expert_ids: Dict[int, set],
    seen_experts: int,
    device: torch.device,
) -> torch.Tensor:
    seen_experts = max(int(seen_experts), 1)
    fallback_eid = seen_experts - 1
    out = []
    for lbl in labels.detach().cpu().tolist():
        cls_id = int(lbl)
        candidates = sorted(int(e) for e in class_to_expert_ids.get(cls_id, set()) if int(e) < seen_experts)
        out.append(candidates[0] if len(candidates) > 0 else fallback_eid)
    return torch.tensor(out, dtype=torch.long, device=device)


def evaluate_fly_bottleneck_detailed(
    model,
    test_loader,
    exposed_class_ids,
    num_classes,
    device,
    class_to_expert_ids: Dict[int, set],
    seen_experts: int,
) -> Dict[str, Tuple[float, np.ndarray]]:
    """Evaluate Fly bottleneck attribution under three routing/head conditions.

    Modes:
    - normal: default RP-router + method head/ensemble.
    - oracle_router: oracle expert id (from class-to-expert mapping) + method head.
    - oracle_router_head: same oracle routing set, and for each sample choose the
      expert that maximizes the true-class logit (oracle head upper bound).
    """
    if not hasattr(model, "infer_with_expert_ids"):
        return {}

    mode_names = ["normal", "oracle_router", "oracle_router_head"]
    stats = {
        m: {
            "correct": 0,
            "total": 0,
            "cls_correct": np.zeros(num_classes, dtype=np.float64),
            "cls_total": np.zeros(num_classes, dtype=np.float64),
        }
        for m in mode_names
    }

    model.eval()
    smoke_max_eval_batches = _resolve_smoke_max_eval_batches()
    eval_batches = 0

    with torch.no_grad():
        for images, labels, _ in test_loader:
            eval_batches += 1
            images = images.to(device)
            labels = labels.to(device)

            logits_normal = model(images, test=True, all_test=False)
            exposed = _align_exposed_with_logits(list(exposed_class_ids), int(logits_normal.shape[1]))
            if not exposed:
                continue

            effective_dim = min(int(logits_normal.shape[1]), len(exposed))
            if effective_dim <= 0:
                continue

            exposed = exposed[:effective_dim]
            exposed_set = set(exposed)
            label_map = {int(cls_id): idx for idx, cls_id in enumerate(exposed)}
            valid_mask = torch.tensor([int(lbl.item()) in exposed_set for lbl in labels], dtype=torch.bool, device=device)
            if not torch.any(valid_mask):
                continue

            labels_valid = labels[valid_mask]
            logits_normal = logits_normal[:, :effective_dim]

            # Oracle router logits: force expert by label->expert mapping.
            oracle_eids = _build_oracle_expert_ids_from_labels(
                labels=labels,
                class_to_expert_ids=class_to_expert_ids,
                seen_experts=seen_experts,
                device=device,
            )
            logits_oracle_router = model.infer_with_expert_ids(images, oracle_eids)[:, :effective_dim]

            # Oracle head logits: evaluate all seen experts once, then choose per-sample
            # the expert with the highest true-class logit within the class candidate set.
            per_expert_logits = []
            for eid in range(max(int(seen_experts), 1)):
                per_expert_logits.append(model.infer_with_expert_id(images, int(eid))[:, :effective_dim])

            logits_oracle_head = torch.zeros_like(logits_oracle_router)
            for i in range(int(images.size(0))):
                cls_id = int(labels[i].item())
                target_local = label_map.get(cls_id, None)
                if target_local is None:
                    continue
                candidates = sorted(int(e) for e in class_to_expert_ids.get(cls_id, set()) if int(e) < int(seen_experts))
                if len(candidates) == 0:
                    candidates = list(range(max(int(seen_experts), 1)))

                best_e = candidates[0]
                best_score = float(per_expert_logits[best_e][i, target_local].item())
                for eid in candidates[1:]:
                    score = float(per_expert_logits[eid][i, target_local].item())
                    if score > best_score:
                        best_score = score
                        best_e = eid
                logits_oracle_head[i] = per_expert_logits[best_e][i]

            mode_logits = {
                "normal": logits_normal,
                "oracle_router": logits_oracle_router,
                "oracle_router_head": logits_oracle_head,
            }

            for mode_name, logits_m in mode_logits.items():
                preds_local = logits_m.argmax(dim=1)
                preds_global = torch.tensor([exposed[idx] for idx in preds_local.cpu().tolist()], device=device)
                preds_valid = preds_global[valid_mask]

                stats[mode_name]["correct"] += int((preds_valid == labels_valid).sum().item())
                stats[mode_name]["total"] += int(labels_valid.shape[0])

                for j in range(int(labels_valid.shape[0])):
                    cls_id = int(labels_valid[j].item())
                    stats[mode_name]["cls_total"][cls_id] += 1
                    if int(preds_valid[j].item()) == cls_id:
                        stats[mode_name]["cls_correct"][cls_id] += 1

            if smoke_max_eval_batches > 0 and eval_batches >= smoke_max_eval_batches:
                break

    model.train()

    out = {}
    for mode_name in mode_names:
        total = float(stats[mode_name]["total"])
        overall_acc = 100.0 * float(stats[mode_name]["correct"]) / total if total > 0 else 0.0
        cls_acc = np.divide(
            stats[mode_name]["cls_correct"],
            stats[mode_name]["cls_total"],
            out=np.zeros_like(stats[mode_name]["cls_correct"]),
            where=stats[mode_name]["cls_total"] > 0,
        ) * 100.0
        out[mode_name] = (overall_acc, cls_acc)

    return out


def run_gcl(cfg: DictConfig, device: torch.device) -> Dict[str, float]:
    logging.info("Initializing clip_gcl with unified GCL entry")
    run_start_time = time.time()
    run_id = os.path.basename(os.getcwd())

    strict_budget_mode = bool(getattr(cfg, "strict_session_budget", True))
    if int(getattr(cfg, "session_epochs", 1)) != 1:
        logging.info(
            f"[STRICT MODE] session_epochs forced to 1 (was {int(getattr(cfg, 'session_epochs', 1))})"
        )
    cfg.session_epochs = 1

    global_method_name = str(getattr(cfg, "method", "lora")).lower()
    proof_protocol_requested = "na"
    proof_projection_enable_requested = None
    proof_projection_mode_requested = "na"
    proof_protocol_effective = "na"
    proof_projection_enable_effective = None
    proof_projection_mode_effective = "na"
    proof_projection_policy_overridden = False
    proof_projection_policy_reason = "na"
    ewc_protocol_requested = "na"
    ewc_protocol_effective = "na"
    ewc_retention_mode_requested = "na"
    ewc_retention_mode_effective = "na"
    ewc_protocol_policy_overridden = False
    ewc_protocol_policy_reason = "na"
    if global_method_name == "proof":
        proof_protocol_mode = str(getattr(cfg, "proof_protocol_mode", "strict_session_task")).lower()
        proof_protocol_requested = str(proof_protocol_mode)
        proof_projection_enable_requested = bool(getattr(cfg, "proof_projection_enable", True))
        proof_projection_mode_requested = str(getattr(cfg, "proof_projection_mode", "lightweight")).lower()
        proof_projection_enable_effective = bool(proof_projection_enable_requested)
        proof_projection_mode_effective = str(proof_projection_mode_requested)
        proof_projection_policy_reason = "user_authoritative"
        if proof_protocol_mode == "strict_session_task":
            if bool(getattr(cfg, "proof_projection_enable", True)):
                logging.info("[STRICT MODE] PROOF projection disabled")
            cfg.proof_projection_enable = False
            cfg.proof_projection_mode = "disabled"
            proof_projection_enable_effective = False
            proof_projection_mode_effective = "disabled"
            proof_projection_policy_overridden = bool(
                (proof_projection_enable_requested is not False)
                or (proof_projection_mode_requested != "disabled")
            )
            proof_projection_policy_reason = "strict_forces_projection_disabled"
        elif proof_protocol_mode == "faithful_session_task":
            cfg.proof_projection_enable = True
            projection_mode = str(getattr(cfg, "proof_projection_mode", "lightweight")).lower()
            if projection_mode not in {"lightweight", "disabled"}:
                logging.info(
                    f"[PROTOCOL] PROOF faithful_session_task requires bounded refinement; "
                    f"proof_projection_mode={projection_mode} -> lightweight"
                )
                cfg.proof_projection_mode = "lightweight"
            proof_projection_enable_effective = True
            proof_projection_mode_effective = str(getattr(cfg, "proof_projection_mode", "lightweight")).lower()
            proof_projection_policy_overridden = bool(
                (proof_projection_enable_requested is not True)
                or (proof_projection_mode_requested not in {"lightweight", "disabled"})
            )
            proof_projection_policy_reason = "faithful_bounds_projection_mode"
        elif proof_protocol_mode == "paper_full":
            cfg.proof_projection_enable = True
            projection_mode = str(getattr(cfg, "proof_projection_mode", "full")).lower()
            if projection_mode != "full":
                logging.info(
                    f"[PROTOCOL] PROOF paper_full enforces full projection training; "
                    f"proof_projection_mode={projection_mode} -> full"
                )
                cfg.proof_projection_mode = "full"
            proof_projection_enable_effective = True
            proof_projection_mode_effective = "full"
            proof_projection_policy_overridden = bool(
                (proof_projection_enable_requested is not True)
                or (proof_projection_mode_requested != "full")
            )
            proof_projection_policy_reason = "paper_full_forces_full_projection"
        else:
            logging.info(
                f"[PROTOCOL] unknown proof_protocol_mode={proof_protocol_mode}; fallback to strict_session_task"
            )
            cfg.proof_protocol_mode = "strict_session_task"
            cfg.proof_projection_enable = False
            cfg.proof_projection_mode = "disabled"
            proof_protocol_mode = "strict_session_task"
            proof_projection_enable_effective = False
            proof_projection_mode_effective = "disabled"
            proof_projection_policy_overridden = True
            proof_projection_policy_reason = "unknown_protocol_fallback_to_strict"

        proof_protocol_effective = str(getattr(cfg, "proof_protocol_mode", proof_protocol_mode)).lower()
        logging.info(
            "[PROOF POLICY] authoritative=proof_protocol_mode "
            f"requested(protocol={proof_protocol_requested},enable={int(bool(proof_projection_enable_requested))},"
            f"mode={proof_projection_mode_requested}) "
            f"effective(protocol={proof_protocol_effective},enable={int(bool(cfg.proof_projection_enable))},"
            f"mode={str(cfg.proof_projection_mode).lower()}) "
            f"overridden={int(bool(proof_projection_policy_overridden))} "
            f"reason={proof_projection_policy_reason}"
        )
    elif global_method_name in {"ewc", "online_ewc", "ewc_kv"}:
        ewc_protocol_audit = _apply_ewc_protocol_policy(cfg)
        ewc_protocol_requested = str(ewc_protocol_audit["ewc_protocol_requested"])
        ewc_protocol_effective = str(ewc_protocol_audit["ewc_protocol_effective"])
        ewc_retention_mode_requested = str(ewc_protocol_audit["ewc_retention_requested"])
        ewc_retention_mode_effective = str(ewc_protocol_audit["ewc_retention_effective"])
        ewc_protocol_policy_overridden = bool(ewc_protocol_audit["ewc_protocol_policy_overridden"])
        ewc_protocol_policy_reason = str(ewc_protocol_audit["ewc_protocol_policy_reason"])

    prompt_profile = None
    prompt_resolution = None
    prompt_profile_overrides_applied = None
    prompt_primary_loss_requested = "na"
    prompt_primary_loss_effective = "na"
    lwf_primary_loss_requested = "na"
    lwf_primary_loss_effective = "na"
    if _is_prompt_family(global_method_name):
        prompt_resolution_raw = _resolved_prompt_audit(cfg, global_method_name)
        prompt_profile = _resolve_prompt_profile(cfg)
        profile_overrides = _prompt_profile_overrides(global_method_name, prompt_profile)
        prompt_profile_overrides_applied = bool(getattr(cfg, "prompt_apply_profile_overrides", True))
        if prompt_profile_overrides_applied:
            _apply_prompt_overrides(cfg, profile_overrides)
        else:
            logging.info("[PROMPT PROFILE] profile overrides disabled by prompt_apply_profile_overrides=0")
        _apply_prompt_protocol_policy(cfg)

        prompt_primary_loss_requested = str(getattr(cfg, "prompt_primary_loss", "clip_pair")).lower()
        prompt_primary_loss_effective = _normalize_prompt_primary_loss(prompt_primary_loss_requested)
        cfg.prompt_primary_loss = prompt_primary_loss_effective

        prompt_resolution_effective = _resolved_prompt_audit(cfg, global_method_name)
        prompt_resolution = {
            "raw": prompt_resolution_raw,
            "effective": prompt_resolution_effective,
            "profile_overrides_applied": int(prompt_profile_overrides_applied),
            "prompt_primary_loss_requested": str(prompt_primary_loss_requested),
            "prompt_primary_loss_effective": str(prompt_primary_loss_effective),
        }
        logging.info(f"[PROMPT PROFILE] resolved={json.dumps(prompt_resolution, sort_keys=True)}")
    elif global_method_name == "lwf":
        lwf_primary_loss_requested = str(getattr(cfg, "lwf_primary_loss", "ce")).lower()
        lwf_primary_loss_effective = _normalize_lwf_primary_loss(lwf_primary_loss_requested)
        cfg.lwf_primary_loss = lwf_primary_loss_effective
        logging.info(
            f"[LWF LOSS] primary loss={lwf_primary_loss_effective} "
            f"(requested={lwf_primary_loss_requested}, effective={lwf_primary_loss_effective})"
        )

    _, clip_transform = clip.load(cfg.model_name, device=device.type, jit=False)
    train_dataset, class_names = get_dataset_for_gcl(cfg, is_train=True, clip_transform=clip_transform)
    test_dataset, _ = get_dataset_for_gcl(cfg, is_train=False, clip_transform=clip_transform)

    num_classes = len(class_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    proof_protocol_mode = str(getattr(cfg, "proof_protocol_mode", "strict_session_task")).lower()
    proof_task_construction_mode = str(
        getattr(cfg, "proof_task_construction_mode", "source_class_incremental")
    ).lower()
    proof_source_faithful = bool(
        global_method_name == "proof"
        and proof_protocol_mode == "paper_full"
        and proof_task_construction_mode in {"source", "source_class_incremental"}
    )

    # Keep method constructors compatible; switch task construction for source-faithful PROOF paper_full.
    cfg.scenario = "class"
    cfg.class_order = list(range(num_classes))
    source_increments = None
    source_train_scenario = None
    source_test_scenario = None

    if proof_source_faithful:
        init_cls, increment, source_increments = _resolve_source_increments(num_classes, cfg)
        cfg.initial_increment = int(init_cls)
        cfg.increment = int(increment)
        cfg.task_num = int(len(source_increments))
        logging.info(
            "[PROOF][paper_full] source-faithful task construction enabled | "
            f"mode={proof_task_construction_mode} initial_increment={int(init_cls)} "
            f"increment={int(increment)} tasks={int(len(source_increments))}"
        )
    else:
        cfg.task_num = int(getattr(cfg, "gcl_sessions", 1))
        cfg.initial_increment = 1
        cfg.increment = 1

    if _is_prompt_family(global_method_name):
        _validate_prompt_hard_session_capacity(cfg, global_method_name)

    model = load_model(cfg, device)
    model.classes_names = class_names
    seq_lora_analyzer = None
    if bool(getattr(cfg, "enable_seq_lora_analysis", False)):
        if _supports_diagnostic_analysis(global_method_name):
            analysis_output_dir = os.path.join(
                os.getcwd(),
                str(getattr(cfg, "analysis_output_dir_name", "analysis_seq_lora")),
            )
            seq_lora_analyzer = SeqLoRAAnalyzer(
                args=cfg,
                model=model,
                device=device,
                output_dir=analysis_output_dir,
                class_names=class_names,
            )
            logging.info(f"[SeqLoRAAnalysis] enabled; output_dir={analysis_output_dir}")
        else:
            logging.warning(
                "[SeqLoRAAnalysis] enable_seq_lora_analysis=true but method is not in the "
                "diagnostic-analysis allowlist; "
                f"skip analysis for method={global_method_name}"
            )
    trainable_summary = _trainable_param_summary(model)
    logging.info(
        "[TRAINABLE PARAMS] "
        f"total={trainable_summary['total_params']} "
        f"trainable={trainable_summary['trainable_params']} "
        f"visual_trainable={trainable_summary['visual_trainable_params']} "
        f"text_trainable={trainable_summary['text_trainable_params']} "
        f"logit_scale_trainable={trainable_summary['logit_scale_trainable_params']} "
        f"freeze_text_encoder={int(bool(getattr(cfg, 'freeze_text_encoder', False)))}"
    )

    online_train_dataset = OnlineIterDataset(train_dataset, iteration=1)
    online_test_dataset = OnlineIterDataset(test_dataset, iteration=1)

    train_sampler = None
    if proof_source_faithful:
        source_train_scenario, _ = build_cl_scenarios(cfg, is_train=True, transforms=model.transforms)
        source_test_scenario, _ = build_cl_scenarios(cfg, is_train=False, transforms=model.transforms)
        session_class_plan = _build_source_session_plan(cfg.class_order, source_increments)
        num_sessions = int(len(session_class_plan))
    else:
        stream_seed = int(getattr(cfg, "stream_seed", cfg.seed))
        train_sampler = OnlineSampler(
            data_source=online_train_dataset,
            num_tasks=int(cfg.gcl_sessions),
            m=int(cfg.gcl_blurry_ratio),
            n=int(cfg.gcl_disjoint_ratio),
            rnd_seed=stream_seed,
            cur_iter=0,
            varing_NM=False,
        )
        session_class_plan = _build_session_class_plan(train_sampler, int(cfg.gcl_sessions))
        num_sessions = int(cfg.gcl_sessions)
        locked_stream_manifest = getattr(cfg, "locked_stream_manifest", None)
        if locked_stream_manifest:
            with open(str(locked_stream_manifest), "r", encoding="utf-8") as handle:
                locked_stream = json.load(handle)
            actual_indices = [list(map(int, values)) for values in train_sampler.indices]
            if actual_indices != locked_stream.get("session_sample_indices"):
                raise ValueError("generated stream differs from locked CUB stream manifest")
            if session_class_plan != locked_stream.get("session_class_plan"):
                raise ValueError("generated class plan differs from locked CUB stream manifest")

    class_intro_session = {}
    class_to_expert_ids = defaultdict(set)
    for sid, classes in enumerate(session_class_plan):
        for cid in classes:
            if int(cid) not in class_intro_session:
                class_intro_session[int(cid)] = sid
            class_to_expert_ids[int(cid)].add(int(sid))
    if hasattr(model, "class_ids_per_task"):
        model.class_ids_per_task = session_class_plan

    metrics = GCLMetrics(num_classes=num_classes, num_sessions=int(num_sessions))

    aux_loss_coeff = _resolve_aux_loss_coeff(cfg, global_method_name)
    logging.info(
        "UnifiedTrainDefaults | "
        f"method={global_method_name} optimizer={str(getattr(cfg, 'optimizer', 'adamw')).lower()} "
        f"scheduler={str(getattr(cfg, 'scheduler', 'cosine')).lower()} "
        f"lr={float(cfg.lr):.3e} wd={float(getattr(cfg, 'weight_decay', 0.0)):.3e} "
        f"batch={int(cfg.train_batch_size)} session_epochs={int(getattr(cfg, 'session_epochs', 1))} "
        f"warmup_ratio={float(getattr(cfg, 'warmup_ratio', 0.0)):.3f} "
        f"min_lr_ratio={float(getattr(cfg, 'min_lr_ratio', 0.0)):.3f} "
        f"grad_clip_norm={float(getattr(cfg, 'grad_clip_norm', 0.0)):.3f} "
        f"label_smoothing={float(getattr(cfg, 'label_smoothing', 0.0)):.3f} "
        f"strict_budget_mode={int(strict_budget_mode)} "
        f"aux_loss_coeff={aux_loss_coeff:.3f}"
    )

    samples_cnt = 0
    next_eval_at = int(cfg.eval_every_n_batches)
    previous_best_old_acc = None
    pre_acc_history = []
    post_acc_history = []
    pre_cls_history = []
    post_cls_history = []
    fly_attr_history = []
    total_main_updates = 0
    total_after_updates = 0
    total_lwf_kd_loss_sum = 0.0
    total_lwf_kd_batches = 0
    total_lwf_teacher_active_batches = 0

    proof_replay_memory = None
    proof_sample_replay_memory = None
    if global_method_name == "proof" and str(getattr(cfg, "proof_protocol_mode", "strict_session_task")).lower() == "paper_full":
        if proof_source_faithful:
            proof_sample_replay_memory = _ProofSampleReplayMemory(
                memory_size=int(getattr(cfg, "proof_memory_size", 2000)),
                memory_per_class=int(getattr(cfg, "proof_memory_per_class", 20)),
                fixed_memory=bool(getattr(cfg, "proof_fixed_memory", False)),
            )
            logging.info(
                "[PROOF][paper_full] source sample replay enabled | "
                f"memory_size={int(getattr(cfg, 'proof_memory_size', 2000))} "
                f"memory_per_class={int(getattr(cfg, 'proof_memory_per_class', 20))} "
                f"fixed_memory={int(bool(getattr(cfg, 'proof_fixed_memory', False)))}"
            )
        else:
            proof_replay_memory = _ProofIndexReplayMemory(
                memory_size=int(getattr(cfg, "proof_memory_size", 2000)),
                memory_per_class=int(getattr(cfg, "proof_memory_per_class", 20)),
                fixed_memory=bool(getattr(cfg, "proof_fixed_memory", False)),
                seed=int(getattr(cfg, "seed", 0)),
            )
            logging.info(
                "[PROOF][paper_full] index replay enabled | "
                f"memory_size={int(getattr(cfg, 'proof_memory_size', 2000))} "
                f"memory_per_class={int(getattr(cfg, 'proof_memory_per_class', 20))} "
                f"fixed_memory={int(bool(getattr(cfg, 'proof_fixed_memory', False)))}"
            )

    with open(cfg.log_path, "w") as f:
        f.write("")

    logging.info(
        f"GCL sessions={int(num_sessions)}, disjoint={cfg.gcl_disjoint_ratio}, blurry={cfg.gcl_blurry_ratio}"
    )

    for session_id in range(int(num_sessions)):
        method_protocol_mode = _resolve_method_protocol_mode(cfg, global_method_name, strict_budget_mode)
        protocol_bucket = _protocol_bucket(method_protocol_mode)
        session_new_classes = session_class_plan[session_id] if session_id < len(session_class_plan) else []
        seen_before = set()
        for prev_id in range(session_id):
            seen_before.update(session_class_plan[prev_id])
        seen_cls = len(seen_before)
        new_cls = len(session_new_classes)
        exposed_cls = len(seen_before.union(set(session_new_classes)))

        logging.info(f"[Run {run_id}] [Dataset {cfg.dataset}] [Protocol {protocol_bucket}]")
        logging.info(
            f"[Session {session_id + 1}/{int(num_sessions)}] SeenCls={seen_cls} "
            f"NewCls={new_cls} Exposed={exposed_cls}"
        )

        if _is_prompt_family(str(getattr(cfg, "method", ""))):
            logging.info(
                "PromptPolicy: "
                f"mode={str(getattr(cfg, 'prompt_window_mode', 'hard_session')).lower()} "
                f"eval_mode={str(getattr(cfg, 'prompt_eval_mode', 'same_as_train')).lower()} "
                f"aux={float(getattr(cfg, 'prompt_aux_loss_coeff', 1.0)):.3f} "
                f"old_access={int(bool(getattr(cfg, 'prompt_train_on_old_classes', True)))} "
                f"mask_old_logits={int(bool(getattr(cfg, 'prompt_mask_old_logits', False)))}"
            )

        model.adaptation(session_id, reset=bool(getattr(cfg, "reset", False)))
        model.train()

        proof_paper_full = bool(global_method_name == "proof" and method_protocol_mode == "paper_full")
        proof_source_stage = bool(proof_paper_full and proof_source_faithful)
        base_session_indices = []
        replay_indices = []
        if proof_source_stage and source_train_scenario is not None:
            source_task_data = source_train_scenario[session_id]
            mem_x, mem_y, mem_t = proof_sample_replay_memory.get() if proof_sample_replay_memory is not None else (None, None, None)
            if mem_x is not None:
                source_task_data.add_samples(mem_x, mem_y, mem_t)
            train_loader = DataLoader(
                source_task_data,
                batch_size=int(cfg.train_batch_size),
                shuffle=True,
                num_workers=int(cfg.num_workers),
                pin_memory=True,
            )
            replay_indices = []
        elif proof_paper_full and proof_replay_memory is not None and train_sampler is not None:
            train_sampler.set_task(session_id)
            base_session_indices = [int(i) for i in train_sampler.indices[session_id]]
            replay_indices = [int(i) for i in proof_replay_memory.get_indices()]
            merged_indices = list(base_session_indices) + list(replay_indices)
            random.Random(int(getattr(cfg, "seed", 0)) + int(session_id)).shuffle(merged_indices)
            train_loader = DataLoader(
                online_train_dataset,
                batch_size=int(cfg.train_batch_size),
                sampler=_FixedIndexSampler(merged_indices),
                num_workers=int(cfg.num_workers),
                pin_memory=True,
            )
            logging.info(
                f"[PROOF][paper_full] session {session_id + 1} train indices | "
                f"base={len(base_session_indices)} replay={len(replay_indices)} merged={len(merged_indices)}"
            )
        else:
            if train_sampler is None:
                raise RuntimeError("train_sampler is unavailable for non-source training path")
            train_sampler.set_task(session_id)
            train_loader = DataLoader(
                online_train_dataset,
                batch_size=int(cfg.train_batch_size),
                sampler=train_sampler,
                num_workers=int(cfg.num_workers),
                pin_memory=True,
            )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = None
        scheduler = None
        warmup_steps = 0
        if len(trainable_params) > 0:
            optimizer = _build_optimizer(trainable_params, cfg)
            total_steps = max(int(getattr(cfg, "session_epochs", 1)) * max(len(train_loader), 1), 1)
            scheduler, warmup_steps = _build_scheduler(optimizer, total_steps, cfg)
            logging.info(
                f"[Session {session_id}] Optimizer={optimizer.__class__.__name__} "
                f"Scheduler={scheduler.__class__.__name__ if scheduler is not None else 'None'} "
                f"TotalSteps={total_steps} WarmupSteps={warmup_steps}"
            )

        skipped_unknown = 0
        session_main_updates = 0
        after_session_updates = 0
        post_session_no_grad_passes = 0
        fisher_samples_seen = 0
        fisher_compute_time = 0.0
        session_end_refine_enable = 0
        session_end_refine_mode = "none"
        session_end_refine_epochs = 0
        session_end_refine_steps = 0
        after_task_info = None
        lr_last = 0.0
        proof_main_projection_steps = 0
        proof_replay_added_indices = int(len(replay_indices)) if proof_paper_full else 0
        proof_replay_memory_size = int(len(replay_indices)) if proof_paper_full else 0
        if proof_source_stage and proof_sample_replay_memory is not None:
            mem_x_now, mem_y_now, _ = proof_sample_replay_memory.get()
            proof_replay_added_indices = int(len(mem_y_now)) if mem_y_now is not None else 0
            proof_replay_memory_size = int(len(mem_y_now)) if mem_y_now is not None else 0
        proof_objective_mode = str(getattr(cfg, "proof_objective_mode", "repo_exact")).lower() if global_method_name == "proof" else "na"
        proof_runtime_task_mode = "na"
        if global_method_name == "proof":
            proof_runtime_task_mode = "source_class_incremental" if proof_source_stage else "session_stream"
        session_lwf_kd_loss_sum = 0.0
        session_lwf_kd_batches = 0
        session_lwf_teacher_active_batches = 0

        method_name = str(getattr(cfg, "method", "lora")).lower()
        is_prompt_method = _is_prompt_family(method_name)
        session_primary_loss_type = "clip_pair"
        if is_prompt_method:
            session_primary_loss_type = str(prompt_primary_loss_effective)
            logging.info(
                f"[PROMPT LOSS] session primary loss={session_primary_loss_type} "
                f"(requested={prompt_primary_loss_requested}, effective={prompt_primary_loss_effective})"
            )
        elif method_name == "lwf":
            session_primary_loss_type = str(lwf_primary_loss_effective)
            logging.info(
                f"[LWF LOSS] session primary loss={session_primary_loss_type} "
                f"(requested={lwf_primary_loss_requested}, effective={lwf_primary_loss_effective})"
            )

        if proof_paper_full:
            logging.info(
                "[PROOF][paper_full] shared CE main-loop skipped; projection stage is treated as session main training"
            )

        smoke_max_train_batches = _resolve_smoke_max_train_batches()
        smoke_skip_after_task = _resolve_smoke_skip_after_task()
        session_train_batches = 0
        session_budget_hit = False
        last_aux_info = {}

        # MindTheGap-style: 计算负类 logit 参考均值（gap_loss_weight=0 或无此方法时 no-op）
        if hasattr(model, "compute_neg_ref") and not proof_paper_full:
            model.compute_neg_ref(
                train_loader,
                device,
                max_batches=int(getattr(cfg, "gap_ref_batches", 5)),
            )

        for epoch in range(int(getattr(cfg, "session_epochs", 1))):
            if proof_paper_full:
                break
            epoch_loss_sum = 0.0
            epoch_loss_steps = 0
            for batch_idx, (images, labels, _) in enumerate(train_loader):
                session_train_batches += 1
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                seen_class_ids = _seen_class_ids_from_model(model, class_to_idx, session_class_plan, session_id)
                exposed = _align_exposed_with_logits(seen_class_ids, int(logits.shape[1]))
                exposed_set = set(exposed)
                label_map = {cls_id: idx for idx, cls_id in enumerate(exposed)}

                valid_mask = torch.tensor([int(lbl.item()) in exposed_set for lbl in labels], dtype=torch.bool, device=device)
                if not torch.any(valid_mask):
                    skipped_unknown += int(labels.shape[0])
                    if smoke_max_train_batches > 0 and session_train_batches >= smoke_max_train_batches:
                        logging.info(
                            f"[Session {session_id}] smoke train batch budget reached "
                            f"({session_train_batches}/{smoke_max_train_batches}); early stop session training"
                        )
                        session_budget_hit = True
                        break
                    continue

                logits_valid = logits[valid_mask]
                labels_valid = labels[valid_mask]
                labels_mapped = torch.tensor([label_map[int(lbl.item())] for lbl in labels_valid], dtype=torch.long, device=device)

                if is_prompt_method:
                    old_dim = int(getattr(model, "_last_valid_out_dim", 0))
                    use_old_mask = bool(getattr(model, "uses_old_class_mask", lambda: False)())
                    if old_dim > 0 and use_old_mask:
                        new_mask = labels_mapped >= old_dim
                        if not torch.any(new_mask):
                            skipped_unknown += int(labels_valid.shape[0])
                            continue
                        logits_valid = logits_valid[new_mask]
                        labels_mapped = labels_mapped[new_mask]
                    if hasattr(model, "register_batch_label_stats"):
                        model.register_batch_label_stats(labels_mapped, old_dim, valid_mask=valid_mask)

                # Apply batch-level logit mask when available (mirrors FlyGCL-main
                # online_train: logit_mask per-batch before CE loss).
                if hasattr(model, "apply_batch_logit_mask"):
                    logits_valid = model.apply_batch_logit_mask(logits_valid, labels_mapped)

                primary_loss = _primary_loss_from_logits(logits_valid, labels_mapped, session_primary_loss_type)
                loss = primary_loss

                aux = model.auxiliary_loss() if hasattr(model, "auxiliary_loss") else None
                if aux is not None:
                    loss = loss + aux_loss_coeff * aux
                last_aux_info = model.auxiliary_info() if hasattr(model, "auxiliary_info") else {}
                if global_method_name == "lwf":
                    kd_value = float(last_aux_info.get("kd", 0.0) or 0.0)
                    teacher_active = bool(last_aux_info.get("teacher_active", False))
                    session_lwf_kd_loss_sum += kd_value
                    session_lwf_kd_batches += 1
                    if teacher_active:
                        session_lwf_teacher_active_batches += 1

                if optimizer is not None and loss.requires_grad:
                    optimizer.zero_grad()
                    if hasattr(model, "accumulate_online_fisher_from_grads"):
                        aux_requires_grad = bool(torch.is_tensor(aux) and aux.requires_grad)
                        # EWC penalty depends only on parameters, not CLIP activations.
                        # Retaining the forward graph doubled peak memory and OOMed
                        # full-CLIP Fisher. Skipping retain_graph here does not change
                        # the CE Fisher or the EWC quadratic gradient.
                        independent_aux = bool(
                            getattr(model, "auxiliary_independent_of_activations", False)
                        )
                        primary_loss.backward(
                            retain_graph=aux_requires_grad and not independent_aux
                        )
                        model.accumulate_online_fisher_from_grads(int(labels_mapped.shape[0]))
                        if aux_requires_grad:
                            (aux_loss_coeff * aux).backward()
                    else:
                        loss.backward()
                    grad_clip_norm = float(getattr(cfg, "grad_clip_norm", 0.0))
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
                    optimizer.step()
                    if hasattr(model, "on_optimizer_step"):
                        model.on_optimizer_step()
                    session_main_updates += 1
                    if scheduler is not None:
                        scheduler.step()

                samples_cnt += int(images.shape[0])
                loss_scalar = float(loss.detach().item())
                epoch_loss_sum += loss_scalar
                epoch_loss_steps += 1
                lr_last = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0

                if batch_idx % int(cfg.log_batch_interval) == 0:
                    lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
                    aux_str = _method_aux_string(last_aux_info, str(getattr(cfg, "method", "lora")))
                    logging.info(
                        f"[Session {session_id}] Epoch [{epoch + 1}/{cfg.session_epochs}] "
                        f"Batch [{batch_idx + 1}/{len(train_loader)}] | "
                        f"Primary({session_primary_loss_type}): {primary_loss.item():.4f} | "
                        f"Aux: {float(aux.detach().item()) if torch.is_tensor(aux) else (float(aux) if aux is not None else 'NA')} | "
                        f"AuxCoeff: {aux_loss_coeff:.3f} | "
                        f"Total: {loss.item():.4f} | "
                        f"LR: {lr:.6e} | "
                        f"SeenCls: {len(exposed)} | "
                        f"SkippedUnknown: {skipped_unknown}"
                        f"{aux_str}"
                    )

                if samples_cnt >= next_eval_at:
                    eval_exposed = _seen_class_ids_from_model(model, class_to_idx, session_class_plan, session_id)
                    eval_sampler = OnlineTestSampler(data_source=online_test_dataset, exposed_class=eval_exposed)
                    eval_loader = DataLoader(
                        online_test_dataset,
                        batch_size=int(cfg.batch_size),
                        sampler=eval_sampler,
                        num_workers=int(cfg.num_workers),
                        pin_memory=True,
                    )
                    eval_acc = evaluate_gcl(model, eval_loader, eval_exposed, device)
                    metrics.add_eval_acc(eval_acc)

                    payload = {
                        "type": "periodic_eval",
                        "session": session_id + 1,
                        "epoch": epoch + 1,
                        "samples": samples_cnt,
                        "accuracy": round(float(eval_acc), 2),
                        "exposed_classes": len(eval_exposed),
                        "skipped_unknown": skipped_unknown,
                    }
                    with open(cfg.log_path, "a") as f:
                        f.write(json.dumps(payload) + "\n")

                    logging.info(
                        f"[Periodic Eval @ {samples_cnt}] Session={session_id + 1} "
                        f"Acc={eval_acc:.2f}% Exposed={len(eval_exposed)}"
                    )
                    next_eval_at += int(cfg.eval_every_n_batches)

                if smoke_max_train_batches > 0 and session_train_batches >= smoke_max_train_batches:
                    logging.info(
                        f"[Session {session_id}] smoke train batch budget reached "
                        f"({session_train_batches}/{smoke_max_train_batches}); early stop session training"
                    )
                    session_budget_hit = True
                    break

            if session_budget_hit:
                break

            elapsed = max(time.time() - run_start_time, 1e-6)
            done_sessions = session_id + 1
            avg_sec_per_session = elapsed / float(done_sessions)
            eta_sec = max(0.0, avg_sec_per_session * float(int(num_sessions) - done_sessions))
            epoch_avg_loss = epoch_loss_sum / max(epoch_loss_steps, 1)
            logging.info(f"[Session {session_id + 1}/{int(num_sessions)}] [Epoch {epoch + 1}/{int(getattr(cfg, 'session_epochs', 1))}]")
            logging.info(
                f"Loss={epoch_avg_loss:.4f} LR={lr_last:.6e} "
                f"MainUpd={session_main_updates} ETA={eta_sec:.1f}s"
            )

        if proof_paper_full and hasattr(model, "after_task"):
            if smoke_skip_after_task:
                logging.info("[SMOKE] skip PROOF after_task stage")
            else:
                after_task_loader = _build_proof_after_task_loader(train_loader, model, class_to_idx)
                model.after_task(after_task_loader)
                if proof_source_stage:
                    samples_cnt += int(len(train_loader.dataset))
                else:
                    samples_cnt += int(len(base_session_indices) + len(replay_indices))
                if hasattr(model, "projection_info"):
                    after_task_info = model.projection_info()
                    projection_steps = int(after_task_info.get("actual_extra_steps", 0))
                    proof_main_projection_steps = int(projection_steps)
                    session_main_updates += projection_steps
                    session_end_refine_enable = int(bool(after_task_info.get("enabled", False)))
                    session_end_refine_mode = str(after_task_info.get("mode", "none"))
                    session_end_refine_epochs = int(after_task_info.get("epochs", 0))
                    session_end_refine_steps = int(after_task_info.get("actual_extra_steps", 0))
                    proof_objective_mode = str(after_task_info.get("objective_mode", proof_objective_mode))

                if proof_source_stage and proof_sample_replay_memory is not None and source_train_scenario is not None:
                    raw_x, raw_y, raw_t = source_train_scenario[session_id].get_raw_samples()
                    total_seen_classes = len(seen_before.union(set(session_new_classes)))
                    proof_sample_replay_memory.add_task(
                        raw_x=raw_x,
                        raw_y=raw_y,
                        raw_t=raw_t,
                        total_classes=total_seen_classes,
                    )
                    _, mem_y_now, _ = proof_sample_replay_memory.get()
                    proof_replay_memory_size = int(len(mem_y_now)) if mem_y_now is not None else 0
                elif proof_replay_memory is not None:
                    total_seen_classes = len(seen_before.union(set(session_new_classes)))
                    proof_replay_memory.update_with_session(
                        session_indices=base_session_indices,
                        targets=online_train_dataset.targets,
                        new_class_ids=session_new_classes,
                        total_seen_classes=total_seen_classes,
                    )
                    proof_replay_memory_size = int(len(proof_replay_memory.get_indices()))

        # v2.1 canonical timeline: evaluate exposed classes before any optional
        # session-end refinement/consolidation stage.
        session_exposed = _seen_class_ids_from_model(model, class_to_idx, session_class_plan, session_id)
        session_eval_sampler_pre = OnlineTestSampler(data_source=online_test_dataset, exposed_class=session_exposed)
        session_eval_loader_pre = DataLoader(
            online_test_dataset,
            batch_size=int(cfg.batch_size),
            sampler=session_eval_sampler_pre,
            num_workers=int(cfg.num_workers),
            pin_memory=True,
        )

        session_acc_pre, cls_acc_pre = evaluate_gcl_detailed(
            model=model,
            test_loader=session_eval_loader_pre,
            exposed_class_ids=session_exposed,
            num_classes=num_classes,
            device=device,
        )
        logging.info(f"[Eval][Session {session_id + 1}] Acc(pre)={session_acc_pre:.2f}")

        if (not proof_paper_full) and hasattr(model, "after_task"):
            if smoke_skip_after_task:
                logging.info("[SMOKE] skip post-session after_task stage")
            else:
                method_name = str(getattr(cfg, "method", "lora")).lower()
                after_task_loader = train_loader
                if method_name == "proof":
                    after_task_loader = _build_proof_after_task_loader(train_loader, model, class_to_idx)
                if strict_budget_mode and method_name in {"ewc", "online_ewc", "ewc_kv"}:
                    after_task_loader = None
                    logging.info("[INFO] Online EWC Fisher was accumulated from in-stream batch gradients")
                model.after_task(after_task_loader)
                if hasattr(model, "projection_info"):
                    after_task_info = model.projection_info()
                    after_session_updates = int(after_task_info.get("actual_extra_steps", 0))
                    session_end_refine_enable = int(bool(after_task_info.get("enabled", False)))
                    session_end_refine_mode = str(after_task_info.get("mode", "none"))
                    session_end_refine_epochs = int(after_task_info.get("epochs", 0))
                    session_end_refine_steps = int(after_task_info.get("actual_extra_steps", 0))
                    proof_objective_mode = str(after_task_info.get("objective_mode", proof_objective_mode))
                if hasattr(model, "post_session_stats"):
                    post_stats = model.post_session_stats()
                    post_session_no_grad_passes = int(post_stats.get("post_session_no_grad_passes", 0))
                    fisher_samples_seen = int(post_stats.get("fisher_samples_seen", 0))
                    fisher_compute_time = float(post_stats.get("fisher_compute_time", 0.0))
                    after_session_updates = max(after_session_updates, int(post_stats.get("after_session_updates", 0)))

        if proof_paper_full:
            # PROOF paper_full has no deferred post-session refinement semantics in this GCL runtime.
            after_session_updates = 0
            session_end_refine_enable = 0
            session_end_refine_mode = "none"
            session_end_refine_epochs = 0
            session_end_refine_steps = 0
            session_acc_post, cls_acc_post = session_acc_pre, cls_acc_pre
        elif after_session_updates > 0:
            session_eval_sampler_post = OnlineTestSampler(data_source=online_test_dataset, exposed_class=session_exposed)
            session_eval_loader_post = DataLoader(
                online_test_dataset,
                batch_size=int(cfg.batch_size),
                sampler=session_eval_sampler_post,
                num_workers=int(cfg.num_workers),
                pin_memory=True,
            )

            session_acc_post, cls_acc_post = evaluate_gcl_detailed(
                model=model,
                test_loader=session_eval_loader_post,
                exposed_class_ids=session_exposed,
                num_classes=num_classes,
                device=device,
            )
        else:
            session_acc_post, cls_acc_post = session_acc_pre, cls_acc_pre

        logging.info(f"[Eval][Session {session_id + 1}] Acc(post)={session_acc_post:.2f}")

        fly_attr_session = None
        method_name_l = str(getattr(cfg, "method", "")).lower()
        fly_attr_enable = bool(getattr(cfg, "fly_attr_enable", True))
        if fly_attr_enable and method_name_l.startswith("fly") and hasattr(model, "infer_with_expert_ids"):
            attr_eval_sampler = OnlineTestSampler(data_source=online_test_dataset, exposed_class=session_exposed)
            attr_eval_loader = DataLoader(
                online_test_dataset,
                batch_size=int(cfg.batch_size),
                sampler=attr_eval_sampler,
                num_workers=int(cfg.num_workers),
                pin_memory=True,
            )
            attr_result = evaluate_fly_bottleneck_detailed(
                model=model,
                test_loader=attr_eval_loader,
                exposed_class_ids=session_exposed,
                num_classes=num_classes,
                device=device,
                class_to_expert_ids=class_to_expert_ids,
                seen_experts=int(session_id + 1),
            )
            if attr_result:
                normal_acc = float(attr_result["normal"][0])
                oracle_router_acc = float(attr_result["oracle_router"][0])
                oracle_head_acc = float(attr_result["oracle_router_head"][0])
                fly_attr_session = {
                    "normal": normal_acc,
                    "oracle_router": oracle_router_acc,
                    "oracle_router_head": oracle_head_acc,
                    "router_gap": oracle_router_acc - normal_acc,
                    "head_gap": oracle_head_acc - oracle_router_acc,
                    "total_gap": oracle_head_acc - normal_acc,
                }
                fly_attr_history.append(dict(fly_attr_session))
                logging.info(
                    f"[Fly Attribution][Session {session_id + 1}] "
                    f"normal={normal_acc:.2f} | oracle_router={oracle_router_acc:.2f} "
                    f"| oracle_router_head={oracle_head_acc:.2f} "
                    f"| router_gap={fly_attr_session['router_gap']:.2f} "
                    f"| head_gap={fly_attr_session['head_gap']:.2f}"
                )

        metrics.add_session_result(session_id, session_acc_post, cls_acc_post)
        pre_acc_history.append(float(session_acc_pre))
        post_acc_history.append(float(session_acc_post))
        pre_cls_history.append(np.asarray(cls_acc_pre, dtype=np.float64))
        post_cls_history.append(np.asarray(cls_acc_post, dtype=np.float64))

        A_avg_pre = float(np.mean(pre_acc_history)) if pre_acc_history else 0.0
        A_last_pre = float(pre_acc_history[-1]) if pre_acc_history else 0.0
        A_avg_post = float(np.mean(post_acc_history)) if post_acc_history else 0.0
        A_last_post = float(post_acc_history[-1]) if post_acc_history else 0.0

        pre_cls_arr = np.stack(pre_cls_history, axis=0)
        post_cls_arr = np.stack(post_cls_history, axis=0)
        forgetting_pre_s, bwt_pre_s, fwt_pre_s = _compute_session_transfer_metrics(
            cls_accs=pre_cls_arr,
            session_plan=session_class_plan,
            class_intro_session=class_intro_session,
            session_id=session_id,
        )
        forgetting_post_s, bwt_post_s, fwt_post_s = _compute_session_transfer_metrics(
            cls_accs=post_cls_arr,
            session_plan=session_class_plan,
            class_intro_session=class_intro_session,
            session_id=session_id,
        )
        session_time_sec = float(time.time() - run_start_time)
        acc_primary = float(session_acc_post) if protocol_bucket == "faithful" else float(session_acc_pre)
        forgetting_s = float(forgetting_post_s) if protocol_bucket == "faithful" else float(forgetting_pre_s)
        bwt_s = float(bwt_post_s) if protocol_bucket == "faithful" else float(bwt_pre_s)
        fwt_s = float(fwt_post_s) if protocol_bucket == "faithful" else float(fwt_pre_s)
        logging.info(
            f"AUC(pre)={float(metrics.compute_A_auc()):.2f} "
            f"Forget(pre)={float(forgetting_pre_s):.2f} BWT(pre)={float(bwt_pre_s):.2f} FWT(pre)={float(fwt_pre_s):.2f}"
        )
        logging.info(
            f"Forget(post)={float(forgetting_post_s):.2f} BWT(post)={float(bwt_post_s):.2f} FWT(post)={float(fwt_post_s):.2f}"
        )
        session_lwf_kd_mean = (
            float(session_lwf_kd_loss_sum) / float(session_lwf_kd_batches)
            if session_lwf_kd_batches > 0 else 0.0
        )
        session_payload = {
            "type": "session_end",
            "session_id": session_id + 1,
            "session": session_id + 1,
            "protocol_mode": protocol_bucket,
            "acc_pre": round(float(session_acc_pre), 2),
            "acc_post": round(float(session_acc_post), 2),
            "acc_primary": round(float(acc_primary), 2),
            "A_avg_pre": round(float(A_avg_pre), 2),
            "A_last_pre": round(float(A_last_pre), 2),
            "A_avg_post": round(float(A_avg_post), 2),
            "A_last_post": round(float(A_last_post), 2),
            "A_auc_pre": round(float(metrics.compute_A_auc()), 4),
            "acc_avg_pre": round(float(A_avg_pre), 4),
            "acc_fin_pre": round(float(A_last_pre), 4),
            "acc_avg_post": round(float(A_avg_post), 4),
            "acc_fin_post": round(float(A_last_post), 4),
            "acc_avg": round(float(A_avg_post if protocol_bucket == "faithful" else A_avg_pre), 4),
            "acc_fin": round(float(A_last_post if protocol_bucket == "faithful" else A_last_pre), 4),
            "forgetting": round(float(forgetting_s), 4),
            "bwt": round(float(bwt_s), 4),
            "fwt": round(float(fwt_s), 4),
            "forgetting_pre": round(float(forgetting_pre_s), 4),
            "bwt_pre": round(float(bwt_pre_s), 4),
            "fwt_pre": round(float(fwt_pre_s), 4),
            "forgetting_post": round(float(forgetting_post_s), 4),
            "bwt_post": round(float(bwt_post_s), 4),
            "fwt_post": round(float(fwt_post_s), 4),
            "accuracy": round(float(session_acc_post), 2),
            "exposed_classes": len(session_exposed),
            "eval_class_count": len(session_exposed),
            "total_samples": samples_cnt,
            "skipped_unknown": skipped_unknown,
            "session_main_updates": int(session_main_updates),
            "after_session_updates": int(after_session_updates),
            "after_task_updates": int(after_session_updates),
            "after_session_refine_deferred": False,
            "refine_for_previous_session": None,
            "refine_trigger_session": None,
            "post_session_no_grad_passes": int(post_session_no_grad_passes),
            "strict_budget_mode": int(strict_budget_mode),
            "method_protocol_mode": method_protocol_mode,
            "prompt_profile": str(prompt_profile) if prompt_profile is not None else None,
            "prompt_profile_overrides_applied": (
                None if prompt_profile_overrides_applied is None else int(bool(prompt_profile_overrides_applied))
            ),
            "prompt_primary_loss_requested": (
                None if prompt_profile is None else str(prompt_primary_loss_requested)
            ),
            "prompt_primary_loss_effective": (
                None if prompt_profile is None else str(prompt_primary_loss_effective)
            ),
            "session_end_refine_enable": int(session_end_refine_enable),
            "session_end_refine_mode": str(session_end_refine_mode),
            "session_end_refine_epochs": int(session_end_refine_epochs),
            "session_end_refine_steps": int(session_end_refine_steps),
            "proof_objective_mode": str(proof_objective_mode),
            "proof_task_construction_mode": str(proof_runtime_task_mode),
            "proof_protocol_mode_requested": str(proof_protocol_requested),
            "proof_protocol_mode_effective": str(proof_protocol_effective),
            "proof_projection_enable_requested": (
                None if proof_projection_enable_requested is None else int(bool(proof_projection_enable_requested))
            ),
            "proof_projection_enable_effective": (
                None if proof_projection_enable_effective is None else int(bool(proof_projection_enable_effective))
            ),
            "proof_projection_mode_requested": str(proof_projection_mode_requested),
            "proof_projection_mode_effective": str(proof_projection_mode_effective),
            "proof_projection_policy_overridden": int(bool(proof_projection_policy_overridden)),
            "proof_projection_policy_reason": str(proof_projection_policy_reason),
            "ewc_protocol_mode_requested": str(ewc_protocol_requested),
            "ewc_protocol_mode_effective": str(ewc_protocol_effective),
            "ewc_retention_mode_requested": str(ewc_retention_mode_requested),
            "ewc_retention_mode_effective": str(ewc_retention_mode_effective),
            "ewc_protocol_policy_overridden": int(bool(ewc_protocol_policy_overridden)),
            "ewc_protocol_policy_reason": str(ewc_protocol_policy_reason),
            "proof_main_projection_steps": int(proof_main_projection_steps),
            "proof_replay_added_indices": int(proof_replay_added_indices),
            "proof_replay_memory_size": int(proof_replay_memory_size),
            "fisher_samples_seen": int(fisher_samples_seen),
            "fisher_compute_time": float(fisher_compute_time),
            "lr_last": float(lr_last),
            "time_sec": round(float(session_time_sec), 4),
        }
        if global_method_name == "lwf":
            session_payload.update({
                "freeze_text_encoder": int(bool(getattr(cfg, "freeze_text_encoder", False))),
                "lwf_primary_loss_requested": str(lwf_primary_loss_requested),
                "lwf_primary_loss_effective": str(lwf_primary_loss_effective),
                "distill_lambda": float(getattr(cfg, "distill_lambda", 1.0)),
                "distill_temp": float(getattr(cfg, "distill_temp", 2.0)),
                "lwf_start_task": int(getattr(cfg, "lwf_start_task", 1)),
                "lwf_kd_mean": round(float(session_lwf_kd_mean), 6),
                "lwf_kd_batches": int(session_lwf_kd_batches),
                "lwf_teacher_active_batches": int(session_lwf_teacher_active_batches),
            })

        if fly_attr_session is not None:
            session_payload.update({
                "fly_attr_normal": round(float(fly_attr_session["normal"]), 4),
                "fly_attr_oracle_router": round(float(fly_attr_session["oracle_router"]), 4),
                "fly_attr_oracle_router_head": round(float(fly_attr_session["oracle_router_head"]), 4),
                "fly_attr_router_gap": round(float(fly_attr_session["router_gap"]), 4),
                "fly_attr_head_gap": round(float(fly_attr_session["head_gap"]), 4),
                "fly_attr_total_gap": round(float(fly_attr_session["total_gap"]), 4),
            })

        if prompt_resolution is not None:
            session_payload["prompt_resolution"] = prompt_resolution

        split_acc = _session_old_new_exposed_acc(cls_acc_post, session_class_plan, session_id)
        session_payload.update(split_acc)

        if seq_lora_analyzer is not None:
            old_analysis_loader = None
            current_analysis_loader = None
            if bool(getattr(cfg, "analysis_record_grad_conflict", False)):
                old_classes_for_analysis = sorted(int(x) for x in seen_before)
                current_classes_for_analysis = sorted(int(x) for x in session_new_classes)
                if old_classes_for_analysis:
                    old_analysis_loader = DataLoader(
                        online_test_dataset,
                        batch_size=int(cfg.batch_size),
                        sampler=OnlineTestSampler(data_source=online_test_dataset, exposed_class=old_classes_for_analysis),
                        num_workers=int(cfg.num_workers),
                        pin_memory=True,
                    )
                if current_classes_for_analysis:
                    current_analysis_loader = DataLoader(
                        online_test_dataset,
                        batch_size=int(cfg.batch_size),
                        sampler=OnlineTestSampler(data_source=online_test_dataset, exposed_class=current_classes_for_analysis),
                        num_workers=int(cfg.num_workers),
                        pin_memory=True,
                    )
            seq_lora_analyzer.run(
                step_idx=session_id,
                task_idx=session_id,
                dataloader=session_eval_loader_pre,
                old_dataloader=old_analysis_loader,
                current_dataloader=current_analysis_loader,
                seen_class_ids=sorted(int(x) for x in seen_before.union(set(session_new_classes))),
                current_class_ids=sorted(int(x) for x in session_new_classes),
                metrics={
                    "old_acc_if_available": split_acc.get("old_exposed_acc", float("nan")),
                    "new_acc_if_available": split_acc.get("new_exposed_acc", float("nan")),
                },
            )

        current_old_acc = split_acc.get("old_exposed_acc", None)
        if current_old_acc is not None:
            prev_best_for_delta = float(previous_best_old_acc) if previous_best_old_acc is not None else float(current_old_acc)
            delta_old_acc = float(current_old_acc) - prev_best_for_delta
            previous_best_old_acc = max(float(previous_best_old_acc), float(current_old_acc)) if previous_best_old_acc is not None else float(current_old_acc)
            session_payload["previous_best_old_acc"] = prev_best_for_delta
            session_payload["current_old_acc"] = float(current_old_acc)
            session_payload["delta_old_acc"] = float(delta_old_acc)

        prompt_method = str(getattr(cfg, "method", "")).lower() in {"l2p", "l2p_official", "dualprompt", "dualprompt_official", "misa", "misa_l2p", "coda"}
        routing = None
        if prompt_method and hasattr(model, "prompt_routing_summary"):
            routing = model.prompt_routing_summary()
            if routing:
                session_payload["prompt_routing"] = routing
            if hasattr(model, "commit_prompt_session_stats"):
                model.commit_prompt_session_stats()

        if after_task_info is not None:
            session_payload["after_task_info"] = after_task_info
        with open(cfg.log_path, "a") as f:
            f.write(json.dumps(session_payload) + "\n")

        logging.info(
            f"Session {session_id + 1} finished | AccPre={session_acc_pre:.2f}% AccPost={session_acc_post:.2f}% "
            f"Exposed={len(session_exposed)} SkippedUnknown={skipped_unknown} "
            f"MainUpdates={session_main_updates} AfterSessionUpdates={after_session_updates} "
            f"PostNoGradPasses={post_session_no_grad_passes} "
            f"OldAcc={split_acc['old_exposed_acc'] if split_acc['old_exposed_acc'] is not None else 'NA'} "
            f"NewAcc={split_acc['new_exposed_acc'] if split_acc['new_exposed_acc'] is not None else 'NA'}"
        )

        if prompt_method and hasattr(model, "prompt_routing_summary"):
            if routing:
                logging.info(
                    f"[PromptRouting][Session {session_id + 1}] mode={routing.get('prompt_window_mode', 'NA')} "
                    f"eval_mode={routing.get('prompt_eval_mode', 'NA')} "
                    f"train_sat={routing.get('train_prompt_saturation', 'NA')} "
                    f"old_sat={routing.get('old_prompt_saturation', 'NA')} "
                    f"new_sat={routing.get('new_prompt_saturation', 'NA')} "
                    f"mixed_sat={routing.get('mixed_prompt_saturation', 'NA')} "
                    f"overlap={routing.get('old_new_prompt_overlap', 'NA')} "
                    f"overlap_ratio={routing.get('old_new_prompt_overlap_ratio', 'NA')} "
                    f"old_revisit={routing.get('old_prompt_revisit_rate', 'NA')} "
                    f"entropy={routing.get('prompt_usage_entropy', 'NA')} "
                    f"top={routing.get('prompt_usage_top', [])} "
                    f"never_used_n={len(routing.get('never_used_prompts', [])) if isinstance(routing.get('never_used_prompts', None), list) else 'NA'}"
                )

        if current_old_acc is not None:
            logging.info(
                f"[ForgettingProxy][Session {session_id + 1}] previous_best_old_acc={session_payload.get('previous_best_old_acc')} "
                f"current_old_acc={session_payload.get('current_old_acc')} "
                f"delta_old_acc={session_payload.get('delta_old_acc')}"
            )
        total_main_updates += int(session_main_updates)
        total_after_updates += int(after_session_updates)
        total_lwf_kd_loss_sum += float(session_lwf_kd_loss_sum)
        total_lwf_kd_batches += int(session_lwf_kd_batches)
        total_lwf_teacher_active_batches += int(session_lwf_teacher_active_batches)

    final_metrics = metrics.get_summary()
    final_forgetting_pre = 0.0
    final_bwt_pre = 0.0
    final_fwt_pre = 0.0
    final_forgetting_post = 0.0
    final_bwt_post = 0.0
    final_fwt_post = 0.0
    if len(pre_cls_history) > 0:
        final_forgetting_pre, final_bwt_pre, final_fwt_pre = _compute_session_transfer_metrics(
            cls_accs=np.stack(pre_cls_history, axis=0),
            session_plan=session_class_plan,
            class_intro_session=class_intro_session,
            session_id=len(pre_cls_history) - 1,
        )
    if len(post_cls_history) > 0:
        final_forgetting_post, final_bwt_post, final_fwt_post = _compute_session_transfer_metrics(
            cls_accs=np.stack(post_cls_history, axis=0),
            session_plan=session_class_plan,
            class_intro_session=class_intro_session,
            session_id=len(post_cls_history) - 1,
        )
    total_lwf_kd_mean = (
        float(total_lwf_kd_loss_sum) / float(total_lwf_kd_batches)
        if total_lwf_kd_batches > 0 else 0.0
    )

    pre_metrics = {
        "type": "pre_refinement_summary",
        "num_sessions": int(len(pre_acc_history)),
        "acc_pre_trajectory": [round(float(x), 2) for x in pre_acc_history],
        "acc_auc": round(float(final_metrics["A_auc"]), 4),
        "acc_auc_pre": round(float(final_metrics["A_auc"]), 4),
        "acc_avg": round(float(np.mean(pre_acc_history)) if pre_acc_history else 0.0, 4),
        "acc_fin": round(float(pre_acc_history[-1]) if pre_acc_history else 0.0, 4),
        "acc_avg_pre": round(float(np.mean(pre_acc_history)) if pre_acc_history else 0.0, 4),
        "acc_fin_pre": round(float(pre_acc_history[-1]) if pre_acc_history else 0.0, 4),
        "forgetting": round(float(final_forgetting_pre), 4),
        "bwt": round(float(final_bwt_pre), 4),
        "fwt": round(float(final_fwt_pre), 4),
        "forgetting_pre": round(float(final_forgetting_pre), 4),
        "bwt_pre": round(float(final_bwt_pre), 4),
        "fwt_pre": round(float(final_fwt_pre), 4),
        "A_avg": round(float(np.mean(pre_acc_history)) if pre_acc_history else 0.0, 2),
        "A_last": round(float(pre_acc_history[-1]) if pre_acc_history else 0.0, 2),
    }
    post_metrics = {
        "type": "post_refinement_summary",
        "num_sessions": int(len(post_acc_history)),
        "acc_post_trajectory": [round(float(x), 2) for x in post_acc_history],
        "acc_auc": round(float(final_metrics["A_auc"]), 4),
        "acc_avg": round(float(np.mean(post_acc_history)) if post_acc_history else 0.0, 4),
        "acc_fin": round(float(post_acc_history[-1]) if post_acc_history else 0.0, 4),
        "acc_avg_post": round(float(np.mean(post_acc_history)) if post_acc_history else 0.0, 4),
        "acc_fin_post": round(float(post_acc_history[-1]) if post_acc_history else 0.0, 4),
        "forgetting": round(float(final_forgetting_post), 4),
        "bwt": round(float(final_bwt_post), 4),
        "fwt": round(float(final_fwt_post), 4),
        "forgetting_post": round(float(final_forgetting_post), 4),
        "bwt_post": round(float(final_bwt_post), 4),
        "fwt_post": round(float(final_fwt_post), 4),
        "A_avg": round(float(np.mean(post_acc_history)) if post_acc_history else 0.0, 2),
        "A_last": round(float(post_acc_history[-1]) if post_acc_history else 0.0, 2),
    }

    with open("metrics_pre_refinement.json", "w") as f:
        pre_metrics.update(analysis_stats_run_metadata())
        json.dump(pre_metrics, f)

    with open("metrics_post_refinement.json", "w") as f:
        post_metrics.update(analysis_stats_run_metadata())
        json.dump(post_metrics, f)

    with open(cfg.log_path, "a") as f:
        f.write(json.dumps({
            "type": "final_summary",
            "A_auc": round(float(final_metrics["A_auc"]), 2),
            "A_avg": round(float(final_metrics["A_avg"]), 2),
            "A_last": round(float(final_metrics["A_last"]), 2),
            "F_last": round(float(final_metrics["F_last"]), 2),
            "BWT_last": round(float(final_metrics["BWT_last"]), 2),
            "forgetting": round(float(final_forgetting_post if protocol_bucket == "faithful" else final_forgetting_pre), 4),
            "bwt": round(float(final_bwt_post if protocol_bucket == "faithful" else final_bwt_pre), 4),
            "fwt": round(float(final_fwt_post if protocol_bucket == "faithful" else final_fwt_pre), 4),
            "forgetting_pre": round(float(final_forgetting_pre), 4),
            "bwt_pre": round(float(final_bwt_pre), 4),
            "fwt_pre": round(float(final_fwt_pre), 4),
            "forgetting_post": round(float(final_forgetting_post), 4),
            "bwt_post": round(float(final_bwt_post), 4),
            "fwt_post": round(float(final_fwt_post), 4),
            "A_avg_pre": pre_metrics["A_avg"],
            "A_last_pre": pre_metrics["A_last"],
            "A_avg_post": post_metrics["A_avg"],
            "A_last_post": post_metrics["A_last"],
            "strict_primary_metric": "A_avg_pre",
            "faithful_primary_metric": "A_avg_post",
            "paper_full_primary_metric": "A_avg_pre",
            "freeze_text_encoder": int(bool(getattr(cfg, "freeze_text_encoder", False))),
            "run_method_tag": str(getattr(cfg, "run_method_tag", global_method_name)),
            "lwf_primary_loss_requested": str(lwf_primary_loss_requested),
            "lwf_primary_loss_effective": str(lwf_primary_loss_effective),
            "distill_lambda": float(getattr(cfg, "distill_lambda", 1.0)) if global_method_name == "lwf" else None,
            "distill_temp": float(getattr(cfg, "distill_temp", 2.0)) if global_method_name == "lwf" else None,
            "lwf_start_task": int(getattr(cfg, "lwf_start_task", 1)) if global_method_name == "lwf" else None,
            "lwf_kd_mean": round(float(total_lwf_kd_mean), 6) if global_method_name == "lwf" else None,
            "lwf_kd_batches": int(total_lwf_kd_batches) if global_method_name == "lwf" else None,
            "lwf_teacher_active_batches": int(total_lwf_teacher_active_batches) if global_method_name == "lwf" else None,
            **analysis_stats_run_metadata(),
        }) + "\n")

    protocol_mode = _resolve_method_protocol_mode(cfg, global_method_name, strict_budget_mode)
    protocol_bucket = _protocol_bucket(protocol_mode)
    final_forgetting = float(final_forgetting_post if protocol_bucket == "faithful" else final_forgetting_pre)
    final_bwt = float(final_bwt_post if protocol_bucket == "faithful" else final_bwt_pre)
    final_fwt = float(final_fwt_post if protocol_bucket == "faithful" else final_fwt_pre)
    runtime_sec = float(time.time() - run_start_time)

    leaderboard_summary = {
        "method": global_method_name,
        "dataset": str(cfg.dataset),
        "protocol_mode": protocol_bucket,
        "seed": int(getattr(cfg, "stream_seed", cfg.seed)),
        "stream_seed": int(getattr(cfg, "stream_seed", cfg.seed)),
        "training_seed": int(cfg.seed),
        "run_id": run_id,
        "status": "success",
        "run_method_tag": str(getattr(cfg, "run_method_tag", global_method_name)),
        "freeze_text_encoder": int(bool(getattr(cfg, "freeze_text_encoder", False))),
        "trainable_param_summary": trainable_summary,
        "primary_metric": "A_avg_post" if protocol_bucket == "faithful" else "A_avg_pre",
        "proof_task_construction_mode": (
            "source_class_incremental"
            if (global_method_name == "proof" and protocol_mode == "paper_full" and proof_source_faithful)
            else ("session_stream" if global_method_name == "proof" else "na")
        ),
        "proof_protocol_mode_requested": str(proof_protocol_requested),
        "proof_protocol_mode_effective": str(proof_protocol_effective),
        "proof_projection_enable_requested": (
            None if proof_projection_enable_requested is None else int(bool(proof_projection_enable_requested))
        ),
        "proof_projection_enable_effective": (
            None if proof_projection_enable_effective is None else int(bool(proof_projection_enable_effective))
        ),
        "proof_projection_mode_requested": str(proof_projection_mode_requested),
        "proof_projection_mode_effective": str(proof_projection_mode_effective),
        "proof_projection_policy_overridden": int(bool(proof_projection_policy_overridden)),
        "proof_projection_policy_reason": str(proof_projection_policy_reason),
        "ewc_protocol_mode_requested": str(ewc_protocol_requested),
        "ewc_protocol_mode_effective": str(ewc_protocol_effective),
        "ewc_retention_mode_requested": str(ewc_retention_mode_requested),
        "ewc_retention_mode_effective": str(ewc_retention_mode_effective),
        "ewc_protocol_policy_overridden": int(bool(ewc_protocol_policy_overridden)),
        "ewc_protocol_policy_reason": str(ewc_protocol_policy_reason),
        "acc_auc": round(float(final_metrics["A_auc"]), 4),
        "acc_avg": round(float(post_metrics["A_avg"] if protocol_bucket == "faithful" else pre_metrics["A_avg"]), 4),
        "acc_fin": round(float(post_metrics["A_last"] if protocol_bucket == "faithful" else pre_metrics["A_last"]), 4),
        "forgetting": round(final_forgetting, 4),
        "bwt": round(final_bwt, 4),
        "fwt": round(float(final_fwt), 4),
        "acc_avg_pre": round(float(pre_metrics["A_avg"]), 4),
        "acc_fin_pre": round(float(pre_metrics["A_last"]), 4),
        "acc_avg_post": round(float(post_metrics["A_avg"]), 4),
        "acc_fin_post": round(float(post_metrics["A_last"]), 4),
        "forgetting_pre": round(float(final_forgetting_pre), 4),
        "bwt_pre": round(float(final_bwt_pre), 4),
        "fwt_pre": round(float(final_fwt_pre), 4),
        "forgetting_post": round(float(final_forgetting_post), 4),
        "bwt_post": round(float(final_bwt_post), 4),
        "fwt_post": round(float(final_fwt_post), 4),
        "runtime_sec": round(runtime_sec, 4),
        "session_main_updates": int(total_main_updates),
        "after_session_updates": int(total_after_updates),
    }
    if global_method_name == "lwf":
        leaderboard_summary.update({
            "lwf_primary_loss_requested": str(lwf_primary_loss_requested),
            "lwf_primary_loss_effective": str(lwf_primary_loss_effective),
            "distill_lambda": float(getattr(cfg, "distill_lambda", 1.0)),
            "distill_temp": float(getattr(cfg, "distill_temp", 2.0)),
            "lwf_start_task": int(getattr(cfg, "lwf_start_task", 1)),
            "lwf_kd_mean": round(float(total_lwf_kd_mean), 6),
            "lwf_kd_batches": int(total_lwf_kd_batches),
            "lwf_teacher_active_batches": int(total_lwf_teacher_active_batches),
        })
    if len(fly_attr_history) > 0:
        last_attr = fly_attr_history[-1]
        leaderboard_summary.update({
            "fly_attr_normal_fin": round(float(last_attr["normal"]), 4),
            "fly_attr_oracle_router_fin": round(float(last_attr["oracle_router"]), 4),
            "fly_attr_oracle_router_head_fin": round(float(last_attr["oracle_router_head"]), 4),
            "fly_attr_router_gap_fin": round(float(last_attr["router_gap"]), 4),
            "fly_attr_head_gap_fin": round(float(last_attr["head_gap"]), 4),
            "fly_attr_total_gap_fin": round(float(last_attr["total_gap"]), 4),
            "fly_attr_router_gap_avg": round(float(np.mean([x["router_gap"] for x in fly_attr_history])), 4),
            "fly_attr_head_gap_avg": round(float(np.mean([x["head_gap"] for x in fly_attr_history])), 4),
            "fly_attr_total_gap_avg": round(float(np.mean([x["total_gap"] for x in fly_attr_history])), 4),
        })
    if prompt_resolution is not None:
        leaderboard_summary["prompt_profile"] = str(prompt_profile)
        leaderboard_summary["prompt_profile_overrides_applied"] = int(bool(prompt_profile_overrides_applied))
        leaderboard_summary["prompt_primary_loss_requested"] = str(prompt_primary_loss_requested)
        leaderboard_summary["prompt_primary_loss_effective"] = str(prompt_primary_loss_effective)
        leaderboard_summary["prompt_resolution"] = prompt_resolution
    leaderboard_summary.update(analysis_stats_run_metadata())
    with open("leaderboard_summary.json", "w") as f:
        json.dump(leaderboard_summary, f, indent=2)

    torch.save(model.state_dict(), "final_loraclip_baseline.pth")

    logging.info("=" * 46)
    logging.info("================ FINAL SUMMARY ================")
    logging.info(f"Method={global_method_name}")
    logging.info(f"Protocol={protocol_bucket}")
    logging.info("")
    logging.info(f"AccAUC={float(final_metrics['A_auc']):.4f}")
    logging.info(f"AccAvg={float(post_metrics['A_avg'] if protocol_bucket == 'faithful' else pre_metrics['A_avg']):.4f}")
    logging.info(f"AccFin={float(post_metrics['A_last'] if protocol_bucket == 'faithful' else pre_metrics['A_last']):.4f}")
    logging.info("")
    logging.info(f"Forget={float(final_forgetting):.4f}")
    logging.info(f"BWT={float(final_bwt):.4f}")
    logging.info(f"FWT={float(final_fwt):.4f}")
    logging.info("")
    logging.info(f"MainUpd={int(total_main_updates)}")
    logging.info(f"AfterUpd={int(total_after_updates)}")
    logging.info(f"Runtime={float(runtime_sec):.4f}")
    logging.info("===============================================")
    return final_metrics


@hydra.main(config_path="configs", config_name="cifar100/flygcl", version_base="1.1")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    _resolve_cfg_paths(cfg)
    _ensure_seq_lora_analysis_defaults(cfg)

    # Hydra changes into the run directory before calling main().  Resolve a
    # relative data root against the directory from which the command was
    # launched so release configs never depend on an author's machine path.
    if cfg.dataset_root and not os.path.isabs(str(cfg.dataset_root)):
        cfg.dataset_root = os.path.abspath(
            os.path.join(hydra.utils.get_original_cwd(), str(cfg.dataset_root))
        )

    seed_everything(int(cfg.seed))

    cfg.workdir = utils.get_workdir(path=hydra.utils.get_original_cwd())
    if not cfg.dataset_root:
        cfg.dataset_root = os.path.join(cfg.workdir, "data")

    if not os.path.isabs(cfg.log_path):
        cfg.log_path = os.path.abspath(cfg.log_path)
    log_dir = os.path.dirname(cfg.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        force=True,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("train.log", mode="w"),
        ],
    )
    utils.save_config(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run. No GPU available.")
    device = torch.device("cuda")
    logging.info(f"Using device: {device}")
    try:
        run_gcl(cfg, device)
    except Exception as e:
        failed_summary = {
            "method": str(getattr(cfg, "method", "unknown")).lower(),
            "dataset": str(getattr(cfg, "dataset", "unknown")),
            "protocol_mode": _protocol_bucket(_resolve_method_protocol_mode(cfg, str(getattr(cfg, "method", "unknown")), bool(getattr(cfg, "strict_session_budget", True)))),
            "seed": int(getattr(cfg, "stream_seed", getattr(cfg, "seed", 0))),
            "stream_seed": int(getattr(cfg, "stream_seed", getattr(cfg, "seed", 0))),
            "training_seed": int(getattr(cfg, "seed", 0)),
            "run_id": os.path.basename(os.getcwd()),
            "status": "failed",
            "reason": str(e),
        }
        failed_summary.update(analysis_stats_run_metadata())
        with open("leaderboard_summary.json", "w") as f:
            json.dump(failed_summary, f, indent=2)
        raise


if __name__ == "__main__":
    main()
