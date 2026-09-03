from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .io_utils import extract_batch, limit_batches
from .grouping_utils import build_old_new_groups
from .io_utils import warn_once


def _clip_core(model):
    return getattr(model, "model", getattr(model, "clip_model", model))


def _prompt_method_features(model, images: torch.Tensor):
    if not all(hasattr(model, name) for name in ("feat", "prompt", "_project_visual")):
        return None
    with torch.no_grad():
        tokens, _ = model.feat(images)
        q = tokens[:, 0, :]
        out, _ = model.feat(
            images,
            prompt=getattr(model, "prompt", None),
            q=q,
            train=False,
            task_id=getattr(model, "current_task", None),
        )
        return model._project_visual(out[:, 0, :])


def extract_image_features(model, images: torch.Tensor) -> torch.Tensor:
    prompt_features = _prompt_method_features(model, images)
    if prompt_features is not None:
        return F.normalize(prompt_features.float(), dim=-1)

    core = _clip_core(model)
    if hasattr(core, "encode_image"):
        features = core.encode_image(images.type(getattr(core, "dtype", images.dtype)))
    elif hasattr(model, "encode_image"):
        features = model.encode_image(images)
    else:
        out = model(images, test=True, return_feature=True) if callable(model) else None
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            features = out[1]
        else:
            raise RuntimeError("model does not expose encode_image or return image features")
    return F.normalize(features.float(), dim=-1)


def collect_diagnostic_features(model, dataloader, device, max_batches: int):
    features = []
    labels = []
    sample_ids = []
    auto_idx = 0
    for _, batch in limit_batches(dataloader, max_batches):
        images, batch_labels, indices = extract_batch(batch)
        if images is None or batch_labels is None:
            continue
        images = images.to(device)
        batch_labels = batch_labels.detach().cpu().long()
        feats = extract_image_features(model, images).detach().cpu()
        features.append(feats)
        labels.append(batch_labels)
        if indices is None:
            ids = torch.arange(auto_idx, auto_idx + int(feats.shape[0]))
            auto_idx += int(feats.shape[0])
        else:
            ids = indices.detach().cpu().long() if torch.is_tensor(indices) else torch.tensor(indices, dtype=torch.long)
        sample_ids.append(ids)
    if not features:
        return None, None, None
    return torch.cat(features, dim=0), torch.cat(labels, dim=0), torch.cat(sample_ids, dim=0)


def _prototype_dict(features: torch.Tensor, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
    groups = defaultdict(list)
    for feat, label in zip(features, labels):
        groups[int(label.item())].append(feat)
    out = {}
    for label, vals in groups.items():
        out[label] = F.normalize(torch.stack(vals, dim=0).mean(dim=0), dim=0).detach().cpu()
    return out


def compute_feature_drift_stats(
    model,
    dataloader,
    reference_features: Optional[Dict[int, torch.Tensor]],
    reference_prototypes: Optional[Dict[int, torch.Tensor]],
    device,
    max_batches: int,
    enable_groupwise: bool = False,
    seen_class_ids=None,
    current_class_ids=None,
    fill_missing_reference_with_current: bool = False,
) -> Tuple[List[Dict], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[str, float]]:
    features, labels, sample_ids = collect_diagnostic_features(model, dataloader, device, max_batches)
    if features is None:
        raise RuntimeError("no diagnostic image features could be extracted")

    current_features = {
        int(sample_id.item()): feat.detach().cpu()
        for sample_id, feat in zip(sample_ids, features)
    }
    current_prototypes = _prototype_dict(features, labels)

    if reference_features is None:
        reference_features = current_features
    if reference_prototypes is None:
        reference_prototypes = current_prototypes

    group_masks = build_old_new_groups(labels, seen_class_ids, current_class_ids) if enable_groupwise else {"all": torch.ones_like(labels, dtype=torch.bool)}
    rows = []
    for group_name in ["all", "old_all", "new_current"]:
        group_mask = group_masks.get(group_name)
        if group_mask is None:
            continue
        if not bool(group_mask.any()):
            warn_once(f"feature_drift group {group_name} is empty; skip row")
            continue
        group_indices = torch.nonzero(group_mask, as_tuple=False).view(-1).tolist()
        group_sample_ids = {int(sample_ids[idx].item()) for idx in group_indices}
        group_labels = labels[group_mask]
        group_features = features[group_mask]

        drift_values = []
        for sample_id, feat in current_features.items():
            if int(sample_id) not in group_sample_ids:
                continue
            ref = reference_features.get(sample_id)
            if ref is None or tuple(ref.shape) != tuple(feat.shape):
                if not fill_missing_reference_with_current:
                    continue
                ref = feat
            drift = 1.0 - float(F.cosine_similarity(feat, ref, dim=0).item())
            drift_values.append(drift)

        proto_drifts = []
        group_prototypes = _prototype_dict(group_features, group_labels)
        for label, proto in group_prototypes.items():
            ref_proto = reference_prototypes.get(label)
            if ref_proto is None or tuple(ref_proto.shape) != tuple(proto.shape):
                if not fill_missing_reference_with_current:
                    continue
                ref_proto = proto
            proto_drifts.append(1.0 - float(F.cosine_similarity(proto, ref_proto, dim=0).item()))

        rows.append({
            "group_name": group_name,
            "layer_idx_or_final": "final",
            "feature_drift_mean": float(torch.tensor(drift_values).mean().item()) if drift_values else float("nan"),
            "feature_drift_std": float(torch.tensor(drift_values).std(unbiased=False).item()) if drift_values else float("nan"),
            "prototype_drift_mean": float(torch.tensor(proto_drifts).mean().item()) if proto_drifts else float("nan"),
        })

    all_row = next((row for row in rows if row["group_name"] == "all"), rows[0] if rows else {})
    summary = {
        "num_samples": int(features.shape[0]),
        "mean_feature_drift": all_row.get("feature_drift_mean", float("nan")),
        "prototype_drift_mean": all_row.get("prototype_drift_mean", float("nan")),
    }
    return rows, current_features, current_prototypes, summary


def compute_text_feature_drift_stats(
    text_features: torch.Tensor,
    text_class_ids,
    reference_text_features: Optional[Dict[int, torch.Tensor]] = None,
    enable_groupwise: bool = False,
    seen_class_ids=None,
    current_class_ids=None,
    text_tokens=None,
) -> Tuple[List[Dict], Dict[int, torch.Tensor], Dict[str, float]]:
    """Compute drift of CLIP text features across sessions."""
    if text_features is None:
        raise RuntimeError("text features are unavailable")
    features = F.normalize(text_features.detach().float().cpu(), dim=-1)
    class_ids = torch.tensor([int(v) for v in text_class_ids], dtype=torch.long)
    if int(features.shape[0]) != int(class_ids.numel()):
        raise RuntimeError(
            f"text features/class ids mismatch: features={int(features.shape[0])} ids={int(class_ids.numel())}"
        )

    current_features = {
        int(class_id.item()): feat.detach().cpu()
        for class_id, feat in zip(class_ids, features)
    }
    if reference_text_features is None:
        reference_text_features = current_features

    group_masks = (
        build_old_new_groups(class_ids, seen_class_ids, current_class_ids)
        if enable_groupwise
        else {"all": torch.ones_like(class_ids, dtype=torch.bool)}
    )

    rows = []
    for group_name in ["all", "old_all", "new_current"]:
        group_mask = group_masks.get(group_name)
        if group_mask is None:
            continue
        if not bool(group_mask.any()):
            warn_once(f"text_feature_drift group {group_name} is empty; skip row")
            continue

        group_class_ids = class_ids[group_mask].tolist()
        drift_values = []
        for class_id in group_class_ids:
            class_id = int(class_id)
            feat = current_features.get(class_id)
            ref = reference_text_features.get(class_id)
            if feat is None or ref is None or tuple(ref.shape) != tuple(feat.shape):
                continue
            drift_values.append(1.0 - float(F.cosine_similarity(feat, ref, dim=0).item()))

        mean_drift = float(torch.tensor(drift_values).mean().item()) if drift_values else float("nan")
        std_drift = float(torch.tensor(drift_values).std(unbiased=False).item()) if drift_values else float("nan")
        rows.append({
            "encoder_side": "text",
            "group_name": group_name,
            "layer_idx_or_final": "final",
            "feature_drift_mean": mean_drift,
            "feature_drift_std": std_drift,
            "prototype_drift_mean": mean_drift,
            "num_text_classes": int(len(group_class_ids)),
        })

    all_row = next((row for row in rows if row["group_name"] == "all"), rows[0] if rows else {})
    summary = {
        "num_text_classes": int(class_ids.numel()),
        "mean_text_feature_drift": all_row.get("feature_drift_mean", float("nan")),
        "text_prototype_drift_mean": all_row.get("prototype_drift_mean", float("nan")),
    }
    return rows, current_features, summary
