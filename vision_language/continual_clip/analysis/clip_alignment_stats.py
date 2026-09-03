from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .feature_stats import collect_diagnostic_features
from .grouping_utils import build_old_new_groups
from .io_utils import warn_once


def _to_int_set(values):
    if values is None:
        return set()
    if torch.is_tensor(values):
        values = values.detach().cpu().view(-1).tolist()
    return {int(v) for v in values}


def compute_clip_alignment_stats(
    model,
    dataloader,
    text_features: torch.Tensor,
    device,
    max_batches: int,
    label_to_text_index: Optional[Dict[int, int]] = None,
    reference_gap_vec: Optional[torch.Tensor] = None,
    enable_groupwise: bool = False,
    seen_class_ids=None,
    current_class_ids=None,
    return_hard_negative: bool = False,
):
    if text_features is None:
        raise RuntimeError("text_features is unavailable")
    text_features = F.normalize(text_features.detach().float().to(device), dim=-1)
    features, labels, _ = collect_diagnostic_features(model, dataloader, device, max_batches)
    if features is None:
        raise RuntimeError("no diagnostic image features could be extracted")

    image_features = F.normalize(features.float().to(device), dim=-1)
    labels_cpu = labels.detach().cpu().long()
    logits = image_features @ text_features.t()

    pos_by_sample = torch.full((int(labels_cpu.numel()),), float("nan"), dtype=torch.float32)
    max_neg_by_sample = torch.full((int(labels_cpu.numel()),), float("nan"), dtype=torch.float32)
    margin_by_sample = torch.full((int(labels_cpu.numel()),), float("nan"), dtype=torch.float32)
    hard_neg_class_ids = torch.full((int(labels_cpu.numel()),), -1, dtype=torch.long)
    text_to_label = {int(v): int(k) for k, v in label_to_text_index.items()} if label_to_text_index is not None else None
    for row_idx, label in enumerate(labels_cpu.tolist()):
        text_idx = label_to_text_index.get(int(label), None) if label_to_text_index is not None else int(label)
        if text_idx is None or text_idx < 0 or text_idx >= int(text_features.shape[0]):
            continue
        row = logits[row_idx]
        pos = row[text_idx]
        if row.numel() > 1:
            mask = torch.ones(row.numel(), dtype=torch.bool, device=row.device)
            mask[text_idx] = False
            neg_values = row.masked_fill(~mask, float("-inf"))
            neg_max, neg_idx = neg_values.max(dim=0)
            margin = pos - neg_max
            hard_text_idx = int(neg_idx.detach().cpu().item())
            hard_neg_class_ids[row_idx] = int(text_to_label.get(hard_text_idx, hard_text_idx)) if text_to_label is not None else hard_text_idx
        else:
            neg_max = torch.tensor(float("nan"), device=row.device)
            margin = torch.tensor(float("nan"), device=row.device)
        pos_by_sample[row_idx] = float(pos.detach().cpu().item())
        max_neg_by_sample[row_idx] = float(neg_max.detach().cpu().item())
        margin_by_sample[row_idx] = float(margin.detach().cpu().item())

    group_masks = build_old_new_groups(labels_cpu, seen_class_ids, current_class_ids) if enable_groupwise else {"all": torch.ones_like(labels_cpu, dtype=torch.bool)}
    current_classes = _to_int_set(current_class_ids)
    old_classes = _to_int_set(seen_class_ids) - current_classes if seen_class_ids is not None else set()
    reference_by_group = reference_gap_vec if isinstance(reference_gap_vec, dict) else {"all": reference_gap_vec}

    rows = []
    hard_rows = []
    current_gap_vecs = {}
    for group_name in ["all", "old_all", "new_current"]:
        group_mask = group_masks.get(group_name)
        if group_mask is None:
            continue
        if not bool(group_mask.any()):
            warn_once(f"clip_alignment group {group_name} is empty; skip row")
            continue
        valid = group_mask & torch.isfinite(pos_by_sample)
        if not bool(valid.any()):
            warn_once(f"clip_alignment group {group_name} has no valid label/text pairs")
        image_group = image_features[group_mask.to(image_features.device)]
        image_mean = F.normalize(image_group.mean(dim=0), dim=0)
        group_labels = labels_cpu[group_mask].tolist()
        text_indices = []
        for label in group_labels:
            text_idx = label_to_text_index.get(int(label), None) if label_to_text_index is not None else int(label)
            if text_idx is not None and 0 <= int(text_idx) < int(text_features.shape[0]):
                text_indices.append(int(text_idx))
        if text_indices:
            unique_text = torch.tensor(sorted(set(text_indices)), dtype=torch.long, device=text_features.device)
            text_mean = F.normalize(text_features[unique_text].mean(dim=0), dim=0)
        else:
            text_mean = F.normalize(text_features.mean(dim=0), dim=0)
        gap_vec = (image_mean - text_mean).detach().cpu()
        current_gap_vecs[group_name] = gap_vec
        gap_norm = float(torch.linalg.norm(gap_vec.float()).item())
        ref_gap = reference_by_group.get(group_name)
        if ref_gap is None and group_name == "all" and not isinstance(reference_gap_vec, dict):
            ref_gap = reference_gap_vec
        if ref_gap is None:
            ref_gap = gap_vec
        if ref_gap is not None and tuple(ref_gap.shape) == tuple(gap_vec.shape):
            gap_direction_drift = 1.0 - float(F.cosine_similarity(gap_vec.float(), ref_gap.float(), dim=0).item())
        else:
            gap_direction_drift = float("nan")

        pos_t = pos_by_sample[valid]
        margin_t = margin_by_sample[valid]
        neg_t = max_neg_by_sample[valid]
        hard_ids = hard_neg_class_ids[valid]
        hard_valid = hard_ids >= 0
        if bool(hard_valid.any()) and current_classes:
            hard_new = torch.tensor([int(v.item()) in current_classes for v in hard_ids[hard_valid]], dtype=torch.float32)
            hard_neg_is_new_ratio = float(hard_new.mean().item())
        else:
            hard_neg_is_new_ratio = float("nan")
        if bool(hard_valid.any()) and old_classes:
            hard_old = torch.tensor([int(v.item()) in old_classes for v in hard_ids[hard_valid]], dtype=torch.float32)
            hard_neg_is_old_ratio = float(hard_old.mean().item())
        else:
            hard_neg_is_old_ratio = float("nan")

        row = {
            "group_name": group_name,
            "pos_cos_mean": float(pos_t.mean().item()) if bool(valid.any()) else float("nan"),
            "pos_cos_std": float(pos_t.std(unbiased=False).item()) if bool(valid.any()) else float("nan"),
            "margin_mean": float(margin_t.mean().item()) if bool(valid.any()) else float("nan"),
            "margin_std": float(margin_t.std(unbiased=False).item()) if bool(valid.any()) else float("nan"),
            "gap_norm": gap_norm,
            "gap_direction_drift": gap_direction_drift,
            "max_neg_cos_mean": float(neg_t.mean().item()) if bool(valid.any()) else float("nan"),
            "max_neg_cos_std": float(neg_t.std(unbiased=False).item()) if bool(valid.any()) else float("nan"),
            "hard_neg_is_new_ratio": hard_neg_is_new_ratio,
        }
        rows.append(row)
        hard_rows.append({
            "group_name": group_name,
            "num_samples": int(valid.sum().item()),
            "pos_cos_mean": row["pos_cos_mean"],
            "max_neg_cos_mean": row["max_neg_cos_mean"],
            "margin_mean": row["margin_mean"],
            "hard_neg_is_new_ratio": hard_neg_is_new_ratio,
            "hard_neg_is_old_ratio": hard_neg_is_old_ratio,
        })

    all_row = next((row for row in rows if row["group_name"] == "all"), rows[0] if rows else {})
    old_row = next((row for row in rows if row["group_name"] == "old_all"), {})
    new_row = next((row for row in rows if row["group_name"] == "new_current"), all_row)
    summary = {
        "mean_old_pos_cos": old_row.get("pos_cos_mean", float("nan")),
        "mean_new_pos_cos": new_row.get("pos_cos_mean", float("nan")),
        "mean_old_margin": old_row.get("margin_mean", float("nan")),
        "mean_new_margin": new_row.get("margin_mean", float("nan")),
        "modality_gap_norm": all_row.get("gap_norm", float("nan")),
        "modality_gap_drift": all_row.get("gap_direction_drift", float("nan")),
        "num_alignment_samples": int(features.shape[0]),
    }
    gap_payload = current_gap_vecs if enable_groupwise else current_gap_vecs.get("all")
    if return_hard_negative:
        return rows, gap_payload, summary, hard_rows
    return rows, gap_payload, summary
