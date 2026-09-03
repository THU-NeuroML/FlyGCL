import math
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn.functional as F


PROTOTYPE_GEOMETRY_FIELDS = [
    "method",
    "seed",
    "step_idx",
    "group_name",
    "num_classes",
    "prototype_drift_mean",
    "prototype_matrix_drift",
    "intra_class_distance",
    "inter_class_distance",
    "intra_inter_ratio",
    "nn_consistency",
]


def _as_tensor(value):
    if torch.is_tensor(value):
        return value.detach().float()
    return torch.tensor(value, dtype=torch.float32)


def compute_prototypes(features, labels) -> Dict[int, torch.Tensor]:
    feats = F.normalize(_as_tensor(features), dim=-1).cpu()
    labels_t = torch.tensor(labels, dtype=torch.long).cpu()
    out = {}
    for class_id in sorted(set(int(v) for v in labels_t.tolist())):
        mask = labels_t == int(class_id)
        if mask.any():
            out[int(class_id)] = F.normalize(feats[mask].mean(dim=0), dim=0)
    return out


def _mean_pairwise_cos_distance(matrix: torch.Tensor) -> float:
    if matrix.shape[0] < 2:
        return math.nan
    sims = F.normalize(matrix, dim=-1) @ F.normalize(matrix, dim=-1).t()
    mask = ~torch.eye(sims.shape[0], dtype=torch.bool)
    return float((1.0 - sims[mask]).mean().item())


def _nearest_neighbors(protos: Dict[int, torch.Tensor], topk: int):
    classes = sorted(protos)
    if len(classes) < 2:
        return {}
    mat = torch.stack([protos[c] for c in classes], dim=0)
    sims = F.normalize(mat, dim=-1) @ F.normalize(mat, dim=-1).t()
    out = {}
    k = min(int(topk), len(classes) - 1)
    for idx, cls in enumerate(classes):
        row = sims[idx].clone()
        row[idx] = -float("inf")
        nn = torch.topk(row, k=k).indices.tolist()
        out[cls] = {classes[int(i)] for i in nn}
    return out


def compute_prototype_geometry(
    features,
    labels,
    reference_prototypes: Optional[Dict[int, torch.Tensor]] = None,
    topk=5,
):
    feats = F.normalize(_as_tensor(features), dim=-1).cpu()
    labels_t = torch.tensor(labels, dtype=torch.long).cpu()
    protos = compute_prototypes(feats, labels_t)
    classes = sorted(protos)
    proto_mat = torch.stack([protos[c] for c in classes], dim=0) if classes else torch.empty(0)

    drift_values = []
    matrix_drift = math.nan
    nn_consistency = math.nan
    if reference_prototypes:
        common = [c for c in classes if c in reference_prototypes]
        for cls in common:
            drift_values.append(1.0 - float(F.cosine_similarity(
                protos[cls].float(), reference_prototypes[cls].float(), dim=0
            ).item()))
        if common:
            cur = torch.stack([protos[c] for c in common], dim=0)
            ref = torch.stack([reference_prototypes[c].float().cpu() for c in common], dim=0)
            matrix_drift = float(torch.linalg.norm(cur - ref).item() / max(len(common), 1))
            cur_nn = _nearest_neighbors({c: protos[c] for c in common}, topk)
            ref_nn = _nearest_neighbors({c: reference_prototypes[c].float().cpu() for c in common}, topk)
            overlaps = []
            for cls in common:
                denom = max(len(cur_nn.get(cls, set())), 1)
                overlaps.append(len(cur_nn.get(cls, set()) & ref_nn.get(cls, set())) / denom)
            nn_consistency = float(torch.tensor(overlaps).mean().item()) if overlaps else math.nan

    intra_values = []
    for cls in classes:
        mask = labels_t == int(cls)
        cls_feats = feats[mask]
        if cls_feats.shape[0] > 0:
            d = 1.0 - (cls_feats @ protos[cls]).mean()
            intra_values.append(float(d.item()))
    intra = float(torch.tensor(intra_values).mean().item()) if intra_values else math.nan
    inter = _mean_pairwise_cos_distance(proto_mat) if classes else math.nan
    ratio = intra / inter if inter and not math.isnan(inter) else math.nan

    return {
        "num_classes": int(len(classes)),
        "prototype_drift_mean": float(torch.tensor(drift_values).mean().item()) if drift_values else math.nan,
        "prototype_matrix_drift": matrix_drift,
        "intra_class_distance": intra,
        "inter_class_distance": inter,
        "intra_inter_ratio": ratio,
        "nn_consistency": nn_consistency,
    }


def empty_prototype_geometry(method_name, seed, step_idx=None, group_name="all"):
    return pd.DataFrame([{
        "method": method_name,
        "seed": seed,
        "step_idx": step_idx if step_idx is not None else math.nan,
        "group_name": group_name,
        "num_classes": math.nan,
        "prototype_drift_mean": math.nan,
        "prototype_matrix_drift": math.nan,
        "intra_class_distance": math.nan,
        "inter_class_distance": math.nan,
        "intra_inter_ratio": math.nan,
        "nn_consistency": math.nan,
    }], columns=PROTOTYPE_GEOMETRY_FIELDS)
