from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .io_utils import extract_batch
from .lora_stats import _matrix_type, _parse_block_idx


def _lora_named_parameters(model):
    return [(name, p) for name, p in model.named_parameters() if "lora" in name.lower() and p.requires_grad]


def _compute_loss(model, batch, device, label_to_logit_index: Optional[Dict[int, int]] = None):
    images, labels, _ = extract_batch(batch)
    if images is None or labels is None:
        raise RuntimeError("batch does not contain images/labels")
    images = images.to(device)
    labels = labels.to(device).long()
    logits = model(images)
    mapped = []
    keep = []
    for idx, label in enumerate(labels.detach().cpu().tolist()):
        target = label_to_logit_index.get(int(label), int(label)) if label_to_logit_index is not None else int(label)
        if 0 <= target < int(logits.shape[1]):
            keep.append(idx)
            mapped.append(target)
    if not keep:
        raise RuntimeError("no labels map into current logits")
    keep_t = torch.tensor(keep, dtype=torch.long, device=device)
    targets = torch.tensor(mapped, dtype=torch.long, device=device)
    return F.cross_entropy(logits.index_select(0, keep_t), targets)


def _collect_grads(model, batch, device, label_to_logit_index=None):
    model.zero_grad(set_to_none=True)
    loss = _compute_loss(model, batch, device, label_to_logit_index=label_to_logit_index)
    loss.backward()
    grads = {}
    for name, param in _lora_named_parameters(model):
        if param.grad is not None:
            grads[name] = param.grad.detach().float().cpu().clone()
    return grads


def compute_lora_gradient_conflict(
    model,
    old_batch,
    current_batch,
    loss_fn=None,
    device=None,
    label_to_logit_index: Optional[Dict[int, int]] = None,
) -> List[Dict]:
    if old_batch is None or current_batch is None:
        raise RuntimeError("old_batch/current_batch are required")
    was_training = model.training
    saved_grads = {name: None if p.grad is None else p.grad.detach().clone() for name, p in model.named_parameters()}
    try:
        model.train()
        old_grads = _collect_grads(model, old_batch, device, label_to_logit_index=label_to_logit_index)
        current_grads = _collect_grads(model, current_batch, device, label_to_logit_index=label_to_logit_index)
    finally:
        model.zero_grad(set_to_none=True)
        for name, p in model.named_parameters():
            grad = saved_grads.get(name)
            p.grad = None if grad is None else grad.to(p.device)
        model.train(was_training)

    rows = []
    old_all = []
    current_all = []
    for name, old_grad in old_grads.items():
        cur_grad = current_grads.get(name)
        if cur_grad is None or tuple(cur_grad.shape) != tuple(old_grad.shape):
            continue
        old_flat = old_grad.flatten()
        cur_flat = cur_grad.flatten()
        rows.append({
            "layer_name": name,
            "matrix_type": _matrix_type(name),
            "grad_cos_old_current": float(F.cosine_similarity(old_flat, cur_flat, dim=0).item()),
            "old_grad_norm": float(torch.linalg.norm(old_flat).item()),
            "current_grad_norm": float(torch.linalg.norm(cur_flat).item()),
        })
        old_all.append(old_flat)
        current_all.append(cur_flat)

    if old_all and current_all:
        old_vec = torch.cat(old_all)
        cur_vec = torch.cat(current_all)
        rows.insert(0, {
            "layer_name": "all_lora",
            "matrix_type": "all",
            "grad_cos_old_current": float(F.cosine_similarity(old_vec, cur_vec, dim=0).item()),
            "old_grad_norm": float(torch.linalg.norm(old_vec).item()),
            "current_grad_norm": float(torch.linalg.norm(cur_vec).item()),
        })
    return rows
