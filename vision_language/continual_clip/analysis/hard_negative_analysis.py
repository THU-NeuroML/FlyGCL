import math
from typing import Iterable, Optional

import pandas as pd
import torch
import torch.nn.functional as F


HARD_NEGATIVE_FIELDS = [
    "method",
    "seed",
    "step_idx",
    "group_name",
    "pos_cos_mean",
    "margin_mean",
    "max_neg_cos_mean",
    "delta_pos_cos",
    "delta_margin",
    "delta_max_neg_cos",
]


def build_hard_negative_summary(clip_alignment_df, method_name, seed):
    if clip_alignment_df is None or clip_alignment_df.empty:
        return pd.DataFrame(columns=HARD_NEGATIVE_FIELDS)
    df = clip_alignment_df.copy()
    for col in ["pos_cos_mean", "margin_mean"]:
        if col not in df.columns:
            df[col] = math.nan
    df["max_neg_cos_mean"] = df["pos_cos_mean"] - df["margin_mean"]
    out = df[["step_idx", "group_name", "pos_cos_mean", "margin_mean", "max_neg_cos_mean"]].copy()
    out.insert(0, "seed", seed)
    out.insert(0, "method", method_name)
    keys = ["group_name"]
    base = out.sort_values("step_idx").groupby(keys, dropna=False).first().reset_index()
    base = base[keys + ["pos_cos_mean", "margin_mean", "max_neg_cos_mean"]].rename(columns={
        "pos_cos_mean": "base_pos_cos",
        "margin_mean": "base_margin",
        "max_neg_cos_mean": "base_max_neg",
    })
    out = out.merge(base, on=keys, how="left")
    out["delta_pos_cos"] = out["pos_cos_mean"] - out["base_pos_cos"]
    out["delta_margin"] = out["margin_mean"] - out["base_margin"]
    out["delta_max_neg_cos"] = out["max_neg_cos_mean"] - out["base_max_neg"]
    return out[HARD_NEGATIVE_FIELDS]


def _as_tensor(value, dtype=None):
    if torch.is_tensor(value):
        return value.detach()
    return torch.tensor(value, dtype=dtype)


def compute_hard_negative_stats(
    image_features,
    text_features,
    labels,
    seen_class_ids: Optional[Iterable[int]] = None,
    old_class_ids: Optional[Iterable[int]] = None,
    new_class_ids: Optional[Iterable[int]] = None,
):
    image = F.normalize(_as_tensor(image_features).float(), dim=-1)
    text = F.normalize(_as_tensor(text_features).float(), dim=-1)
    labels_t = _as_tensor(labels, dtype=torch.long).long().cpu()
    device = image.device
    text = text.to(device)

    if seen_class_ids is None:
        class_ids = list(range(int(text.shape[0])))
    else:
        class_ids = [int(v) for v in seen_class_ids]
    text_index_by_class = {class_id: idx for idx, class_id in enumerate(class_ids)}
    new_set = {int(v) for v in new_class_ids} if new_class_ids is not None else set()

    logits = image @ text.t()
    rows = []
    for row_idx, label in enumerate(labels_t.tolist()):
        class_id = int(label)
        text_idx = text_index_by_class.get(class_id)
        if text_idx is None or text_idx >= logits.shape[1]:
            rows.append({
                "pos_cos": math.nan,
                "max_neg_cos": math.nan,
                "margin": math.nan,
                "hard_neg_class_id": math.nan,
                "hard_neg_is_new": math.nan,
                "hard_neg_similarity": math.nan,
                "correct_similarity": math.nan,
            })
            continue
        row = logits[row_idx].detach().cpu()
        pos = float(row[text_idx].item())
        neg = row.clone()
        neg[text_idx] = -float("inf")
        hard_idx = int(torch.argmax(neg).item())
        hard_cls = class_ids[hard_idx] if hard_idx < len(class_ids) else hard_idx
        hard_sim = float(row[hard_idx].item())
        rows.append({
            "pos_cos": pos,
            "max_neg_cos": hard_sim,
            "margin": pos - hard_sim,
            "hard_neg_class_id": int(hard_cls),
            "hard_neg_is_new": int(hard_cls in new_set) if new_class_ids is not None else math.nan,
            "hard_neg_similarity": hard_sim,
            "correct_similarity": pos,
        })
    return pd.DataFrame(rows)
