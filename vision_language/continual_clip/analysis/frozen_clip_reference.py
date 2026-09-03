import math

import pandas as pd
import torch
import torch.nn.functional as F

from .feature_stats import extract_image_features
from .grouping_utils import build_old_new_groups
from .io_utils import extract_batch, limit_batches, safe_to_csv, warn_once


FROZEN_CLIP_ALIGNMENT_FIELDS = [
    "step_idx",
    "task_idx",
    "group_name",
    "num_samples",
    "frozen_pos_cos",
    "frozen_max_neg_cos",
    "frozen_margin",
    "frozen_acc",
    "frozen_gap_norm",
    "pos_cos_mean",
    "max_neg_cos_mean",
    "margin_mean",
    "acc",
    "gap_norm",
]


def _text_index_for_label(label, label_to_text_index, class_ids):
    if label_to_text_index:
        return label_to_text_index.get(int(label))
    if class_ids is not None:
        try:
            return class_ids.index(int(label))
        except ValueError:
            return None
    return int(label)


def _class_for_text_index(text_idx, text_to_label, class_ids):
    if text_to_label is not None:
        return int(text_to_label.get(int(text_idx), int(text_idx)))
    if class_ids is not None and 0 <= int(text_idx) < len(class_ids):
        return int(class_ids[int(text_idx)])
    return int(text_idx)


def run_frozen_clip_reference(
    model,
    dataloader,
    text_features,
    seen_class_ids,
    device,
    output_path,
    step_idx=0,
    task_idx=None,
    current_class_ids=None,
    label_to_text_index=None,
    max_batches=-1,
):
    """
    Compute frozen-CLIP alignment reference for an explicit diagnostic run.

    This function is intentionally not called from training. Callers must pass a
    frozen model, dataloader, and text features explicitly.
    """
    if model is None or dataloader is None or text_features is None:
        warn_once("frozen CLIP reference missing model/dataloader/text_features; writing NaN schema")
        safe_to_csv(pd.DataFrame(columns=FROZEN_CLIP_ALIGNMENT_FIELDS), output_path)
        return

    class_ids = [int(v) for v in seen_class_ids] if seen_class_ids is not None else None
    text = F.normalize(text_features.detach().float().to(device), dim=-1)
    text_to_label = {int(v): int(k) for k, v in label_to_text_index.items()} if label_to_text_index else None

    image_chunks, label_chunks = [], []
    with torch.no_grad():
        for _, batch in limit_batches(dataloader, max_batches):
            images, labels, _ = extract_batch(batch)
            if images is None or labels is None:
                continue
            image_chunks.append(extract_image_features(model, images.to(device)).detach().cpu())
            label_chunks.append(labels.detach().cpu().long() if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.long))
    if not image_chunks:
        warn_once("frozen CLIP reference found no image/label batches; writing NaN schema")
        safe_to_csv(pd.DataFrame(columns=FROZEN_CLIP_ALIGNMENT_FIELDS), output_path)
        return

    image_features = F.normalize(torch.cat(image_chunks, dim=0).float().to(device), dim=-1)
    labels = torch.cat(label_chunks, dim=0).long()
    logits = image_features @ text.t()
    preds = logits.argmax(dim=1).detach().cpu()
    pos_by_sample = torch.full((int(labels.numel()),), float("nan"), dtype=torch.float32)
    max_neg_by_sample = torch.full_like(pos_by_sample, float("nan"))
    margin_by_sample = torch.full_like(pos_by_sample, float("nan"))
    correct_by_sample = torch.full_like(pos_by_sample, float("nan"))

    for row_idx, label in enumerate(labels.tolist()):
        text_idx = _text_index_for_label(label, label_to_text_index, class_ids)
        if text_idx is None or int(text_idx) < 0 or int(text_idx) >= int(logits.shape[1]):
            continue
        row = logits[row_idx].detach().cpu()
        pos = float(row[int(text_idx)].item())
        neg = row.clone()
        neg[int(text_idx)] = -float("inf")
        max_neg = float(neg.max().item()) if int(row.numel()) > 1 else math.nan
        pred_class = _class_for_text_index(int(preds[row_idx]), text_to_label, class_ids)
        pos_by_sample[row_idx] = pos
        max_neg_by_sample[row_idx] = max_neg
        margin_by_sample[row_idx] = pos - max_neg
        correct_by_sample[row_idx] = float(int(pred_class == int(label)))

    if current_class_ids is not None:
        group_masks = build_old_new_groups(labels, seen_class_ids, current_class_ids)
    else:
        group_masks = {"all": torch.ones_like(labels, dtype=torch.bool)}

    rows = []
    for group_name in ["all", "old_all", "new_current"]:
        group_mask = group_masks.get(group_name)
        if group_mask is None:
            continue
        if not bool(group_mask.any()):
            warn_once(f"frozen CLIP reference group {group_name} is empty; skip row")
            continue
        valid = group_mask & torch.isfinite(pos_by_sample)
        pos = pos_by_sample[valid]
        neg = max_neg_by_sample[valid]
        margin = margin_by_sample[valid]
        correct = correct_by_sample[valid]
        if bool(group_mask.any()):
            image_mean = F.normalize(image_features[group_mask.to(image_features.device)].mean(dim=0), dim=0)
            text_indices = []
            for label in labels[group_mask].tolist():
                text_idx = _text_index_for_label(label, label_to_text_index, class_ids)
                if text_idx is not None and 0 <= int(text_idx) < int(text.shape[0]):
                    text_indices.append(int(text_idx))
            text_mean = F.normalize(text[torch.tensor(sorted(set(text_indices)), dtype=torch.long, device=text.device)].mean(dim=0), dim=0) if text_indices else F.normalize(text.mean(dim=0), dim=0)
            frozen_gap_norm = float(torch.linalg.norm((image_mean - text_mean).float()).item())
        else:
            frozen_gap_norm = math.nan
        row = {
            "step_idx": step_idx,
            "task_idx": task_idx if task_idx is not None else step_idx,
            "group_name": group_name,
            "num_samples": int(valid.sum().item()),
            "frozen_pos_cos": float(pos.mean().item()) if bool(valid.any()) else math.nan,
            "frozen_max_neg_cos": float(neg.mean().item()) if bool(valid.any()) else math.nan,
            "frozen_margin": float(margin.mean().item()) if bool(valid.any()) else math.nan,
            "frozen_acc": float(correct.mean().item()) if bool(valid.any()) else math.nan,
            "frozen_gap_norm": frozen_gap_norm,
        }
        row.update({
            "pos_cos_mean": row["frozen_pos_cos"],
            "max_neg_cos_mean": row["frozen_max_neg_cos"],
            "margin_mean": row["frozen_margin"],
            "acc": row["frozen_acc"],
            "gap_norm": row["frozen_gap_norm"],
        })
        rows.append(row)
    safe_to_csv(pd.DataFrame(rows, columns=FROZEN_CLIP_ALIGNMENT_FIELDS), output_path)
