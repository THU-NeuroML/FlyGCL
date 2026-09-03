import math
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .feature_stats import extract_image_features
from .grouping_utils import build_old_new_groups
from .io_utils import extract_batch, limit_batches
from .io_utils import warn_once
from .lora_stats import _parse_block_idx


CLASS_REFERENCE_PREFIX = "__class__"


def _entropy(p: torch.Tensor) -> float:
    p = p.float()
    p = p / p.sum().clamp_min(1e-12)
    return float((-(p * torch.log(p.clamp_min(1e-12))).sum()).item())


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.float() / p.float().sum().clamp_min(1e-12)
    q = q.float() / q.float().sum().clamp_min(1e-12)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (torch.log(p.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum()
    js = js + 0.5 * (q * (torch.log(q.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum()
    return float(js.item())


def _topk_overlap(p: torch.Tensor, q: torch.Tensor, k: int = 10) -> float:
    if p.numel() == 0 or q.numel() == 0:
        return float("nan")
    k = min(int(k), int(p.numel()), int(q.numel()))
    if k <= 0:
        return float("nan")
    a = set(torch.topk(p.float(), k=k).indices.cpu().tolist())
    b = set(torch.topk(q.float(), k=k).indices.cpu().tolist())
    return float(len(a.intersection(b)) / max(k, 1))


def _attention_distance(p: torch.Tensor) -> float:
    n = int(p.numel())
    side = int(math.sqrt(n))
    if side * side != n:
        return float("nan")
    coords = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), dim=-1).reshape(-1, 2)
    center = torch.tensor([(side - 1) / 2.0, (side - 1) / 2.0])
    dist = torch.linalg.norm(coords.float() - center.float(), dim=1)
    probs = p.float() / p.float().sum().clamp_min(1e-12)
    return float((probs.cpu() * dist).sum().item())


def _cls_to_patch(weights: torch.Tensor) -> Optional[torch.Tensor]:
    if weights.ndim == 3:
        patch_count = max(int(weights.shape[1]) - 1, 0)
        if patch_count <= 0 or int(weights.shape[2]) < patch_count:
            return None
        return weights[:, 0, -patch_count:]
    if weights.ndim == 4:
        patch_count = max(int(weights.shape[2]) - 1, 0)
        if patch_count <= 0 or int(weights.shape[3]) < patch_count:
            return None
        return weights[:, :, 0, -patch_count:].mean(dim=1)
    return None


def _eot_to_tokens(weights: torch.Tensor, text_tokens: Optional[torch.Tensor], input_prompt_len: int = 0) -> Optional[torch.Tensor]:
    if text_tokens is None or not torch.is_tensor(text_tokens):
        return None
    if weights.ndim == 4:
        attn = weights.mean(dim=1)
    elif weights.ndim == 3:
        attn = weights
    else:
        return None
    if attn.ndim != 3:
        return None
    batch = min(int(attn.shape[0]), int(text_tokens.shape[0]))
    if batch <= 0:
        return None
    attn = attn[:batch]
    tokens = text_tokens[:batch].detach().cpu()
    input_prompt_len = max(int(input_prompt_len), 0)
    expected_prompted_len = int(text_tokens.shape[1]) + input_prompt_len
    has_input_prompt = input_prompt_len > 0 and int(attn.shape[1]) >= expected_prompted_len and int(attn.shape[2]) >= expected_prompted_len
    eot_pos = tokens.argmax(dim=-1).long()
    query_eot_pos = eot_pos + (input_prompt_len if has_input_prompt else 0)
    query_eot_pos = query_eot_pos.clamp(min=0, max=int(attn.shape[1]) - 1)
    rows = []
    positions = torch.arange(int(attn.shape[-1]))
    for idx in range(batch):
        row = attn[idx, int(query_eot_pos[idx].item()), :].float().cpu()
        valid = positions <= int(query_eot_pos[idx].item())
        masked = torch.zeros_like(row)
        masked[valid] = row[valid]
        rows.append(masked / masked.sum().clamp_min(1e-12))
    return torch.stack(rows, dim=0)


def _reference_from_class_means(
    reference_attention: Dict,
    layer_name: str,
    class_ids,
    target_shape,
) -> Optional[torch.Tensor]:
    refs = []
    for class_id in sorted({int(v) for v in class_ids}):
        ref = reference_attention.get((CLASS_REFERENCE_PREFIX, class_id, layer_name))
        if ref is not None and tuple(ref.shape) == tuple(target_shape):
            refs.append(ref.float())
    if not refs:
        return None
    ref = torch.stack(refs, dim=0).mean(dim=0)
    return ref / ref.sum().clamp_min(1e-12)


def _patch_attention_modules(model, captured: Dict[str, Dict[str, List[torch.Tensor]]]):
    patched = []
    core = getattr(model, "model", getattr(model, "clip_model", model))
    for name, module in core.named_modules():
        if not hasattr(module, "attn"):
            continue
        modality = "vision" if "visual" in name else "text"
        if modality == "text" and "transformer" not in name:
            continue
        original_forward = module.attn.forward

        def make_forward(module_name, module_modality, original_method):
            def forward(self, query, key, value, *args, **kwargs):
                kwargs["need_weights"] = True
                out, weights = original_method(query, key, value, *args, **kwargs)
                if weights is not None:
                    captured.setdefault(module_modality, {}).setdefault(module_name, []).append(weights.detach().float().cpu())
                return out, weights
            return types.MethodType(forward, module.attn)

        module.attn.forward = make_forward(name, modality, original_forward)
        patched.append((module.attn, original_forward))
    return patched


def _capture_lengths(captured: Dict[str, Dict[str, List[torch.Tensor]]]) -> Dict[str, Dict[str, int]]:
    return {
        modality: {name: len(values) for name, values in layers.items()}
        for modality, layers in captured.items()
    }


def _trim_capture_to(captured: Dict[str, Dict[str, List[torch.Tensor]]], lengths: Dict[str, Dict[str, int]]) -> None:
    for modality, layers in captured.items():
        modality_lengths = lengths.get(modality, {})
        for name, values in layers.items():
            keep = int(modality_lengths.get(name, 0))
            if len(values) > keep:
                del values[keep:]


def _is_input_patch_prompt(prompt_inject: str) -> bool:
    return str(prompt_inject).lower() in {
        "patch",
        "patch_prompt",
        "input",
        "input_prompt",
        "input_patch_prompt",
    }


def _prompt_input_len(prompt_module) -> int:
    if prompt_module is None:
        return 0
    return int(getattr(prompt_module, "prompt_length", 0)) * int(getattr(prompt_module, "top_k", 1))


def _run_text_forward(model, text_tokens: Optional[torch.Tensor], device, captured: Dict[str, Dict[str, List[torch.Tensor]]]) -> int:
    if text_tokens is None:
        return 0
    text_tokens = text_tokens.to(device)

    text_feat = getattr(model, "text_feat", None)
    text_prompt = getattr(model, "text_prompt", None)
    if text_feat is not None and text_prompt is not None:
        with torch.no_grad():
            before_query = _capture_lengths(captured)
            query_tokens, _ = text_feat(text_tokens)
            _trim_capture_to(captured, before_query)
            eot_idx = text_tokens.argmax(dim=-1).long()
            q = query_tokens[torch.arange(query_tokens.shape[0], device=query_tokens.device), eot_idx]
            text_feat(
                text_tokens,
                prompt=text_prompt,
                q=q,
                train=False,
                task_id=getattr(model, "current_task", None),
            )
        inject = getattr(model, "text_prompt_inject", "")
        return _prompt_input_len(text_prompt) if _is_input_patch_prompt(inject) else 0

    core = getattr(model, "model", getattr(model, "clip_model", model))
    with torch.no_grad():
        if hasattr(core, "encode_text"):
            core.encode_text(text_tokens)
        elif hasattr(model, "encode_text"):
            model.encode_text(text_tokens)
    return 0


def compute_attention_stats(
    model,
    dataloader,
    reference_attention: Optional[Dict[str, torch.Tensor]],
    device,
    max_batches: int,
    enable_groupwise: bool = False,
    seen_class_ids=None,
    current_class_ids=None,
    text_tokens: Optional[torch.Tensor] = None,
) -> Tuple[List[Dict], Dict[str, torch.Tensor], Dict[str, float]]:
    captured: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    label_chunks: List[torch.Tensor] = []
    patched = _patch_attention_modules(model, captured)
    if not patched:
        raise RuntimeError("no patchable attention modules found")
    try:
        with torch.no_grad():
            for _, batch in limit_batches(dataloader, max_batches):
                images, labels, _ = extract_batch(batch)
                if images is None:
                    continue
                if labels is not None:
                    label_chunks.append(labels.detach().cpu().long() if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.long))
                extract_image_features(model, images.to(device))
            text_input_prompt_len = _run_text_forward(model, text_tokens, device, captured)
    finally:
        for module, original_forward in patched:
            module.forward = original_forward

    if not captured:
        raise RuntimeError("attention hooks ran but captured no weights")
    labels_all = torch.cat(label_chunks, dim=0) if label_chunks else None
    if enable_groupwise and labels_all is None:
        warn_once("attention group-wise requested but labels are unavailable; use group_name=all")
        enable_groupwise = False

    reference_attention = reference_attention or {}
    current_reference = {}
    rows = []
    drift_vals = []
    for layer_name, weights_list in captured.get("vision", {}).items():
        patches = [_cls_to_patch(weights) for weights in weights_list]
        patches = [patch for patch in patches if patch is not None]
        if not patches:
            continue
        min_patch_count = min(int(patch.shape[-1]) for patch in patches)
        if min_patch_count <= 0:
            continue
        cls_patch = torch.cat([patch[..., -min_patch_count:] for patch in patches], dim=0)
        labels_for_layer = labels_all
        if labels_all is not None and int(labels_all.numel()) != int(cls_patch.shape[0]):
            if int(labels_all.numel()) > 0 and int(cls_patch.shape[0]) % int(labels_all.numel()) == 0:
                repeats = int(cls_patch.shape[0]) // int(labels_all.numel())
                labels_for_layer = torch.cat([chunk.view(-1).repeat(repeats) for chunk in label_chunks], dim=0)
            else:
                warn_once(f"attention labels/weights count mismatch for {layer_name}; use group_name=all")
                labels_for_layer = None
        if enable_groupwise and labels_for_layer is not None and int(labels_for_layer.numel()) == int(cls_patch.shape[0]):
            layer_groups = build_old_new_groups(labels_for_layer, seen_class_ids, current_class_ids)
        else:
            layer_groups = {"all": torch.ones(int(cls_patch.shape[0]), dtype=torch.bool)}

        if labels_for_layer is not None and int(labels_for_layer.numel()) == int(cls_patch.shape[0]):
            for class_id in labels_for_layer.unique().tolist():
                class_mask = labels_for_layer == int(class_id)
                if not bool(class_mask.any()):
                    continue
                class_attn = cls_patch[class_mask].mean(dim=0)
                class_attn = class_attn / class_attn.sum().clamp_min(1e-12)
                current_reference[(CLASS_REFERENCE_PREFIX, int(class_id), layer_name)] = class_attn.detach().cpu()

        for group_name in ["all", "old_all", "new_current"]:
            group_mask = layer_groups.get(group_name)
            if group_mask is None:
                continue
            if not bool(group_mask.any()):
                warn_once(f"attention group {group_name} is empty; skip row")
                continue
            patch_group = cls_patch[group_mask]
            mean_attn = patch_group.mean(dim=0)
            mean_attn = mean_attn / mean_attn.sum().clamp_min(1e-12)
            ref_key = (group_name, layer_name)
            current_reference[ref_key] = mean_attn.detach().cpu()
            ref = None
            if group_name != "all" and labels_for_layer is not None and int(labels_for_layer.numel()) == int(cls_patch.shape[0]):
                group_class_ids = labels_for_layer[group_mask].unique().tolist()
                ref = _reference_from_class_means(reference_attention, layer_name, group_class_ids, mean_attn.shape)
            if ref is None:
                ref = reference_attention.get(ref_key)
            if ref is None and group_name == "all":
                ref = reference_attention.get(layer_name)
            if ref is not None and tuple(ref.shape) == tuple(mean_attn.shape):
                js = _js_divergence(mean_attn, ref)
                overlap = _topk_overlap(mean_attn, ref)
                if group_name == "all":
                    drift_vals.append(js)
            else:
                js = float("nan")
                overlap = float("nan")
            rows.append({
                "encoder_side": "visual",
                "modality": "vision",
                "group_name": group_name,
                "layer_idx": _parse_block_idx(layer_name),
                "head_idx_or_mean": "mean",
                "attention_entropy": _entropy(mean_attn),
                "attention_drift_js": js,
                "topk_overlap": overlap,
                "attention_distance": _attention_distance(mean_attn),
            })

    for layer_name, weights_list in captured.get("text", {}).items():
        token_rows = [_eot_to_tokens(weights, text_tokens, input_prompt_len=text_input_prompt_len) for weights in weights_list]
        token_rows = [row for row in token_rows if row is not None]
        if not token_rows:
            continue
        min_token_count = min(int(row.shape[-1]) for row in token_rows)
        if min_token_count <= 0:
            continue
        eot_token = torch.cat([row[..., :min_token_count] for row in token_rows], dim=0)
        mean_attn = eot_token.mean(dim=0)
        mean_attn = mean_attn / mean_attn.sum().clamp_min(1e-12)
        ref_key = ("text", "all", layer_name)
        current_reference[ref_key] = mean_attn.detach().cpu()
        ref = reference_attention.get(ref_key)
        if ref is not None and tuple(ref.shape) == tuple(mean_attn.shape):
            js = _js_divergence(mean_attn, ref)
            overlap = _topk_overlap(mean_attn, ref)
        else:
            js = float("nan")
            overlap = float("nan")
        rows.append({
            "encoder_side": "text",
            "modality": "text",
            "group_name": "all",
            "layer_idx": _parse_block_idx(layer_name),
            "head_idx_or_mean": "mean",
            "attention_entropy": _entropy(mean_attn),
            "attention_drift_js": js,
            "topk_overlap": overlap,
            "attention_distance": float("nan"),
        })

    summary = {
        "mean_attention_drift": float(torch.tensor(drift_vals).mean().item()) if drift_vals else float("nan"),
    }
    return rows, current_reference, summary
