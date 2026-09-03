from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .grouping_utils import class_partitions, group_for_class
from .io_utils import mean


LEVEL1_STEP_FIELDS = [
    "step", "method", "seed", "dataset", "class_set_mode", "group_name",
    "pos_cos", "max_neg_cos", "margin", "mean_neg_cos", "topk_neg_cos",
    "hard_negative_class_id", "hard_negative_similarity", "hard_negative_group",
    "old_pos_cos", "new_pos_cos", "future_pos_cos",
    "seen_pos_cos", "all_pos_cos",
    "old_margin", "new_margin", "future_margin", "seen_margin", "all_margin",
    "old_max_neg_cos", "new_max_neg_cos", "future_max_neg_cos",
    "seen_max_neg_cos", "all_max_neg_cos",
    "old_to_new_hard_negative_ratio", "old_to_future_hard_negative_ratio",
    "new_to_old_hard_negative_ratio",
    "delta_pos_cos_vs_initial", "delta_margin_vs_initial", "delta_margin_vs_previous", "delta_margin_vs_frozen",
    "delta_max_neg_vs_initial", "delta_pos_cos_vs_frozen", "delta_max_neg_vs_frozen",
    "extra_margin_drop_vs_frozen",
    "extra_old_margin_drop_vs_frozen", "extra_new_margin_drop_vs_frozen", "extra_future_margin_drop_vs_frozen",
]

LEVEL1_SAMPLE_FIELDS = [
    "step", "method", "seed", "dataset", "class_set_mode", "sample_id", "label",
    "group_name", "pos_cos", "max_neg_cos", "margin", "mean_neg_cos",
    "hard_negative_class_id", "hard_negative_similarity", "hard_negative_group",
]

FROZEN_STEP_FIELDS = [
    "step", "dataset", "split", "class_set_mode", "group_name",
    "frozen_old_pos_cos", "frozen_new_pos_cos", "frozen_future_pos_cos",
    "frozen_old_margin", "frozen_new_margin", "frozen_future_margin",
    "frozen_seen_margin", "frozen_all_margin",
    "frozen_old_max_neg_cos", "frozen_new_max_neg_cos", "frozen_future_max_neg_cos",
    "frozen_seen_max_neg_cos", "frozen_all_max_neg_cos",
    "frozen_seen_pos_cos", "frozen_all_pos_cos",
    "frozen_pos_cos", "frozen_max_neg_cos", "frozen_margin", "frozen_mean_neg_cos",
    "hard_negative_class_id", "hard_negative_similarity", "hard_negative_group",
    "frozen_old_to_new_hard_negative_ratio", "frozen_old_to_future_hard_negative_ratio",
    "old_to_new_hard_negative_ratio", "old_to_future_hard_negative_ratio",
    "extra_old_margin_drop_vs_frozen", "extra_new_margin_drop_vs_frozen", "extra_future_margin_drop_vs_frozen",
]

FROZEN_SAMPLE_FIELDS = [
    "step", "dataset", "split", "class_set_mode", "sample_id", "label",
    "group_name", "pos_cos", "max_neg_cos", "margin", "mean_neg_cos",
    "hard_negative_class_id", "hard_negative_similarity", "hard_negative_group",
]


def modes_from_arg(raw: str) -> List[str]:
    mode = str(raw or "both").lower()
    if mode in {"both", "all"}:
        return ["seen_only", "all_classes"]
    if mode in {"seen", "seen_classes_only"}:
        return ["seen_only"]
    if mode in {"all_dataset_classes", "all"}:
        return ["all_classes"]
    return [mode]


def load_config(config_path: Path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(config_path))
    OmegaConf.set_struct(cfg, False)
    return cfg


def prepare_gcl_context(config_path: Path, device: str):
    import clip
    from continual_clip.OnlineIterDataset import OnlineIterDataset
    from continual_clip.datasets import get_dataset_for_gcl
    from continual_clip.utils.onlinesampler import OnlineSampler

    cfg = load_config(config_path)
    cfg.scenario = "class"
    _, transform = clip.load(cfg.model_name, device=device, jit=False)
    train_dataset, class_names = get_dataset_for_gcl(cfg, is_train=True, clip_transform=transform)
    test_dataset, _ = get_dataset_for_gcl(cfg, is_train=False, clip_transform=transform)
    cfg.class_order = list(range(len(class_names)))
    cfg.task_num = int(getattr(cfg, "gcl_sessions", getattr(cfg, "task_num", 1)))
    cfg.initial_increment = 1
    cfg.increment = 1
    online_train = OnlineIterDataset(train_dataset, iteration=1)
    online_test = OnlineIterDataset(test_dataset, iteration=1)
    sampler = OnlineSampler(
        data_source=online_train,
        num_tasks=int(cfg.gcl_sessions),
        m=int(cfg.gcl_blurry_ratio),
        n=int(cfg.gcl_disjoint_ratio),
        rnd_seed=int(cfg.seed),
        cur_iter=0,
        varing_NM=False,
    )
    session_plan = []
    for sid in range(int(cfg.gcl_sessions)):
        disjoint = sampler.disjoint_classes[sid] if sid < len(sampler.disjoint_classes) else []
        blurry = sampler.blurry_classes[sid] if sid < len(sampler.blurry_classes) else []
        session_plan.append(sorted(set(int(x) for x in list(disjoint) + list(blurry))))
    return cfg, class_names, online_test, session_plan


def _select_indices_by_group(targets: Sequence[int], partitions: Mapping[str, set], groups: Sequence[str], limit: int) -> List[int]:
    per_group = defaultdict(list)
    for idx, label in enumerate(targets):
        group = group_for_class(int(label), partitions)
        if group in groups or "all" in groups or "seen" in groups:
            per_group[group].append(int(idx))
        if "seen" in groups and int(label) in partitions.get("seen", set()):
            per_group["seen"].append(int(idx))
        per_group["all"].append(int(idx))

    selected = []
    for group in groups:
        idxs = list(dict.fromkeys(per_group.get(group, [])))
        if limit and limit > 0:
            idxs = idxs[: int(limit)]
        selected.extend(idxs)
    return list(dict.fromkeys(selected))


def _candidate_classes(mode: str, partitions: Mapping[str, set]) -> List[int]:
    if str(mode).lower() == "seen_only":
        return sorted(int(x) for x in partitions.get("seen", set()))
    return sorted(int(x) for x in partitions.get("all", set()))


def _safe_nan() -> float:
    return float("nan")


def _to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return _safe_nan()


def _aggregate(
    rows: Sequence[Mapping[str, object]],
    group: str,
    meta: Mapping[str, object],
    partitions: Optional[Mapping[str, set]] = None,
) -> Dict[str, object]:
    group = str(group)
    if group == "all":
        subset = list(rows)
    elif group == "seen" and partitions is not None:
        seen_classes = set(partitions.get("seen", set()))
        subset = [r for r in rows if int(r.get("label", -1)) in seen_classes]
    else:
        subset = [r for r in rows if str(r.get("group_name")) == group]
    if not subset:
        return {
            **meta,
            "group_name": group,
            "pos_cos": _safe_nan(),
            "max_neg_cos": _safe_nan(),
            "margin": _safe_nan(),
            "mean_neg_cos": _safe_nan(),
            "topk_neg_cos": _safe_nan(),
            "hard_negative_class_id": "",
            "hard_negative_similarity": _safe_nan(),
            "hard_negative_group": "",
        }
    hard_counts = defaultdict(int)
    for r in subset:
        hard_counts[str(r.get("hard_negative_group", ""))] += 1
    hard_group = max(hard_counts.items(), key=lambda x: x[1])[0] if hard_counts else ""
    return {
        **meta,
        "group_name": group,
        "pos_cos": mean(r.get("pos_cos") for r in subset),
        "max_neg_cos": mean(r.get("max_neg_cos") for r in subset),
        "margin": mean(r.get("margin") for r in subset),
        "mean_neg_cos": mean(r.get("mean_neg_cos") for r in subset),
        "topk_neg_cos": _safe_nan(),
        "hard_negative_class_id": "",
        "hard_negative_similarity": mean(r.get("hard_negative_similarity") for r in subset),
        "hard_negative_group": hard_group,
    }


def _ratio(rows: Sequence[Mapping[str, object]], source: str, target: str) -> float:
    subset = [r for r in rows if str(r.get("group_name")) == source]
    if not subset:
        return _safe_nan()
    valid = [r for r in subset if str(r.get("hard_negative_group", ""))]
    if not valid:
        return _safe_nan()
    hits = [r for r in valid if str(r.get("hard_negative_group")) == target]
    return float(len(hits) / len(valid)) if valid else _safe_nan()


def _add_groupwide_fields(step_rows: List[Dict[str, object]], sample_rows: Sequence[Mapping[str, object]]) -> None:
    agg = {str(r.get("group_name")): r for r in step_rows}
    for row in step_rows:
        for g in ("old", "new", "future", "seen", "all"):
            src = agg.get(g, {})
            row[f"{g}_pos_cos"] = src.get("pos_cos", _safe_nan())
            row[f"{g}_margin"] = src.get("margin", _safe_nan())
            row[f"{g}_max_neg_cos"] = src.get("max_neg_cos", _safe_nan())
        row["old_to_new_hard_negative_ratio"] = _ratio(sample_rows, "old", "new")
        row["old_to_future_hard_negative_ratio"] = _ratio(sample_rows, "old", "future")
        row["new_to_old_hard_negative_ratio"] = _ratio(sample_rows, "new", "old")


def _compute_sample_metrics(
    sims,
    labels,
    sample_ids,
    candidate_classes: Sequence[int],
    partitions: Mapping[str, set],
    meta: Mapping[str, object],
) -> List[Dict[str, object]]:
    import torch

    out = []
    class_to_col = {int(c): i for i, c in enumerate(candidate_classes)}
    cand_tensor = torch.tensor([int(c) for c in candidate_classes], device=sims.device, dtype=torch.long)
    for i in range(int(sims.shape[0])):
        label = int(labels[i].item())
        sample_id = int(sample_ids[i].item())
        group = group_for_class(label, partitions)
        if len(candidate_classes) == 0:
            pos = max_neg = margin = mean_neg = hard_sim = _safe_nan()
            hard_id = ""
            hard_group = ""
        else:
            row = sims[i]
            true_col = class_to_col.get(label)
            if true_col is None:
                pos = _safe_nan()
                neg = row
            else:
                pos = float(row[true_col].detach().cpu().item())
                keep = torch.ones(row.shape[0], dtype=torch.bool, device=row.device)
                keep[true_col] = False
                neg = row[keep]
            if neg.numel() == 0:
                max_neg = margin = mean_neg = hard_sim = _safe_nan()
                hard_id = ""
                hard_group = ""
            else:
                hard_local = int(torch.argmax(neg).detach().cpu().item())
                if true_col is None:
                    hard_col = hard_local
                else:
                    neg_cols = torch.arange(row.shape[0], device=row.device)[keep]
                    hard_col = int(neg_cols[hard_local].detach().cpu().item())
                hard_id = int(cand_tensor[hard_col].detach().cpu().item())
                hard_sim = float(row[hard_col].detach().cpu().item())
                max_neg = hard_sim
                mean_neg = float(neg.mean().detach().cpu().item())
                margin = float(pos - max_neg) if not math.isnan(float(pos)) else _safe_nan()
                hard_group = group_for_class(hard_id, partitions)
        out.append({
            **meta,
            "sample_id": sample_id,
            "label": label,
            "group_name": group,
            "pos_cos": pos,
            "max_neg_cos": max_neg,
            "margin": margin,
            "mean_neg_cos": mean_neg,
            "hard_negative_class_id": hard_id,
            "hard_negative_similarity": hard_sim,
            "hard_negative_group": hard_group,
        })
    return out


def _text_features_openai(clip_model, class_names: Sequence[str], template: str, device: str):
    import clip
    import torch.nn.functional as F

    tokens = clip.tokenize([template.format(c) for c in class_names]).to(device)
    text = clip_model.encode_text(tokens).float()
    return F.normalize(text, dim=-1)


def _frozen_features(clip_model, images, text_features, candidate_classes: Sequence[int]):
    import torch.nn.functional as F

    image_features = clip_model.encode_image(images).float()
    image_features = F.normalize(image_features, dim=-1)
    text = text_features[list(candidate_classes)]
    sims = image_features @ text.t()
    return sims


def _method_text_features(method_model, class_names: Sequence[str], template: str, device: str):
    import torch
    import torch.nn.functional as F
    import clip

    if hasattr(method_model, "model") and hasattr(method_model.model, "encode_text"):
        tokens = clip.tokenize([template.format(c) for c in class_names]).to(device)
        text = method_model.model.encode_text(tokens).float()
        return F.normalize(text, dim=-1)
    if hasattr(method_model, "clip_model") and hasattr(method_model.clip_model, "encode_text"):
        tokens = clip.tokenize([template.format(c) for c in class_names]).to(device)
        text = method_model.clip_model.encode_text(tokens).float()
        return F.normalize(text, dim=-1)
    raise RuntimeError(f"Cannot extract text features from method model type={type(method_model)}")


def _method_image_features(method_model, images):
    import torch
    import torch.nn.functional as F

    if hasattr(method_model, "model") and hasattr(method_model.model, "encode_image"):
        feat = method_model.model.encode_image(images).float()
        return F.normalize(feat, dim=-1)
    if all(hasattr(method_model, attr) for attr in ("feat", "prompt", "clip_model")):
        with torch.no_grad():
            tokens, _ = method_model.feat(images)
            q = tokens[:, 0, :]
            out, _ = method_model.feat(
                images,
                prompt=(method_model.prompt if getattr(method_model, "use_vision_prompt", True) else None),
                q=q,
                train=False,
                task_id=getattr(method_model, "current_task", None),
            )
            cls = out[:, 0, :]
            if hasattr(method_model, "_project_visual"):
                feat = method_model._project_visual(cls)
            else:
                feat = cls.float()
            return F.normalize(feat.float(), dim=-1)
    raise RuntimeError(f"Cannot extract image features from method model type={type(method_model)}")


def load_method_model(cfg, class_names: Sequence[str], session_plan: Sequence[Sequence[int]], device: str, checkpoint: Optional[Path], step: int):
    import torch
    from continual_clip.models import load_model

    cfg.class_order = list(range(len(class_names)))
    cfg.task_num = int(len(session_plan))
    cfg.initial_increment = 1
    cfg.increment = 1
    cfg.scenario = "class"
    model = load_model(cfg, torch.device(device))
    model.classes_names = list(class_names)
    if hasattr(model, "class_ids_per_task"):
        model.class_ids_per_task = [list(map(int, x)) for x in session_plan]
    for sid in range(int(step) + 1):
        model.adaptation(sid, reset=False)
    if checkpoint and checkpoint.exists():
        state = torch.load(str(checkpoint), map_location=device)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _make_loader(dataset, selected_indices: Sequence[int], batch_size: int):
    from torch.utils.data import DataLoader, Subset

    subset = Subset(dataset, list(selected_indices))
    return DataLoader(subset, batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=False)


def recompute_frozen_reference(
    config_path: Path,
    dataset_name: str,
    split: str,
    class_set_modes: Sequence[str],
    groups: Sequence[str],
    batch_size: int,
    device: str,
    max_samples_per_group: int,
    max_steps: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    import clip
    import torch

    cfg, class_names, online_test, session_plan = prepare_gcl_context(config_path, device)
    model, _ = clip.load(cfg.model_name, device=device, jit=False)
    model = model.float().eval()
    text_features = _text_features_openai(model, class_names, cfg.prompt_template, device)
    steps = min(len(session_plan), int(max_steps)) if max_steps else len(session_plan)
    step_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for step in range(steps):
            partitions = class_partitions(session_plan, step)
            selected = _select_indices_by_group(online_test.targets, partitions, groups, max_samples_per_group)
            if not selected:
                selected = []
            loader = _make_loader(online_test, selected, batch_size)
            for mode in class_set_modes:
                cand = _candidate_classes(mode, partitions)
                mode_samples = []
                for images, labels, sample_ids in loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    sample_ids = sample_ids.to(device)
                    sims = _frozen_features(model, images, text_features, cand)
                    mode_samples.extend(_compute_sample_metrics(
                        sims, labels, sample_ids, cand, partitions,
                        {"step": step, "dataset": dataset_name, "split": split, "class_set_mode": mode},
                    ))
                sample_rows.extend(mode_samples)
                meta = {"step": step, "dataset": dataset_name, "split": split, "class_set_mode": mode}
                grouped = [_aggregate(mode_samples, g, meta, partitions) for g in groups]
                _add_groupwide_fields(grouped, mode_samples)
                for row in grouped:
                    frozen_row = {
                        **row,
                        "frozen_pos_cos": row.get("pos_cos", _safe_nan()),
                        "frozen_max_neg_cos": row.get("max_neg_cos", _safe_nan()),
                        "frozen_margin": row.get("margin", _safe_nan()),
                        "frozen_mean_neg_cos": row.get("mean_neg_cos", _safe_nan()),
                        "frozen_old_pos_cos": row.get("old_pos_cos", _safe_nan()),
                        "frozen_new_pos_cos": row.get("new_pos_cos", _safe_nan()),
                        "frozen_future_pos_cos": row.get("future_pos_cos", _safe_nan()),
                        "frozen_old_margin": row.get("old_margin", _safe_nan()),
                        "frozen_new_margin": row.get("new_margin", _safe_nan()),
                        "frozen_future_margin": row.get("future_margin", _safe_nan()),
                        "frozen_seen_margin": row.get("seen_margin", _safe_nan()),
                        "frozen_all_margin": row.get("all_margin", _safe_nan()),
                        "frozen_old_max_neg_cos": row.get("old_max_neg_cos", _safe_nan()),
                        "frozen_new_max_neg_cos": row.get("new_max_neg_cos", _safe_nan()),
                        "frozen_future_max_neg_cos": row.get("future_max_neg_cos", _safe_nan()),
                        "frozen_seen_max_neg_cos": row.get("seen_max_neg_cos", _safe_nan()),
                        "frozen_all_max_neg_cos": row.get("all_max_neg_cos", _safe_nan()),
                        "frozen_seen_pos_cos": row.get("seen_pos_cos", _safe_nan()),
                        "frozen_all_pos_cos": row.get("all_pos_cos", _safe_nan()),
                        "frozen_old_to_new_hard_negative_ratio": row.get("old_to_new_hard_negative_ratio", _safe_nan()),
                        "frozen_old_to_future_hard_negative_ratio": row.get("old_to_future_hard_negative_ratio", _safe_nan()),
                        "extra_old_margin_drop_vs_frozen": _safe_nan(),
                        "extra_new_margin_drop_vs_frozen": _safe_nan(),
                        "extra_future_margin_drop_vs_frozen": _safe_nan(),
                    }
                    step_rows.append(frozen_row)
    summary = {
        "dataset": dataset_name,
        "split": split,
        "num_steps": steps,
        "num_step_rows": len(step_rows),
        "num_sample_rows": len(sample_rows),
        "mean_frozen_margin": mean(r.get("frozen_margin") for r in step_rows),
        "status": "real_forward",
    }
    return step_rows, sample_rows, summary


def recompute_method_alignment(
    config_path: Path,
    run_dir: Path,
    method: str,
    dataset_name: str,
    seed: str,
    class_set_modes: Sequence[str],
    groups: Sequence[str],
    batch_size: int,
    device: str,
    max_samples_per_group: int,
    checkpoint: Optional[Path] = None,
    frozen_step_rows: Optional[Sequence[Mapping[str, object]]] = None,
    max_steps: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    import torch

    cfg, class_names, online_test, session_plan = prepare_gcl_context(config_path, device)
    if method:
        cfg.method = method
    ckpt = checkpoint or (run_dir / "final_loraclip_baseline.pth")
    steps = min(len(session_plan), int(max_steps)) if max_steps else len(session_plan)
    frozen_index = {}
    for row in frozen_step_rows or []:
        key = (int(row.get("step", 0)), str(row.get("class_set_mode", "")), str(row.get("group_name", "")))
        frozen_index[key] = row
    step_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for step in range(steps):
            model = load_method_model(cfg, class_names, session_plan, device, ckpt, step)
            text_features = _method_text_features(model, class_names, cfg.prompt_template, device)
            partitions = class_partitions(session_plan, step)
            selected = _select_indices_by_group(online_test.targets, partitions, groups, max_samples_per_group)
            loader = _make_loader(online_test, selected, batch_size)
            for mode in class_set_modes:
                cand = _candidate_classes(mode, partitions)
                mode_samples = []
                for images, labels, sample_ids in loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    sample_ids = sample_ids.to(device)
                    image_features = _method_image_features(model, images)
                    sims = image_features @ text_features[list(cand)].t() if cand else image_features[:, :0]
                    mode_samples.extend(_compute_sample_metrics(
                        sims, labels, sample_ids, cand, partitions,
                        {"step": step, "method": str(cfg.method), "seed": seed, "dataset": dataset_name, "class_set_mode": mode},
                    ))
                sample_rows.extend(mode_samples)
                meta = {"step": step, "method": str(cfg.method), "seed": seed, "dataset": dataset_name, "class_set_mode": mode}
                grouped = [_aggregate(mode_samples, g, meta, partitions) for g in groups]
                _add_groupwide_fields(grouped, mode_samples)
                for row in grouped:
                    frow = frozen_index.get((step, mode, str(row.get("group_name", ""))), {})
                    row["delta_margin_vs_frozen"] = _to_float(row.get("margin")) - _to_float(frow.get("frozen_margin"))
                    row["delta_pos_cos_vs_frozen"] = _to_float(row.get("pos_cos")) - _to_float(frow.get("frozen_pos_cos"))
                    row["delta_max_neg_vs_frozen"] = _to_float(row.get("max_neg_cos")) - _to_float(frow.get("frozen_max_neg_cos"))
                    row["delta_pos_cos_vs_initial"] = _safe_nan()
                    row["delta_margin_vs_initial"] = _safe_nan()
                    row["delta_margin_vs_previous"] = _safe_nan()
                    row["delta_max_neg_vs_initial"] = _safe_nan()
                    row["extra_margin_drop_vs_frozen"] = row["delta_margin_vs_frozen"]
                    row["extra_old_margin_drop_vs_frozen"] = _to_float(row.get("old_margin")) - _to_float(frow.get("frozen_old_margin"))
                    row["extra_new_margin_drop_vs_frozen"] = _to_float(row.get("new_margin")) - _to_float(frow.get("frozen_new_margin"))
                    row["extra_future_margin_drop_vs_frozen"] = _to_float(row.get("future_margin")) - _to_float(frow.get("frozen_future_margin"))
                    step_rows.append(row)
    summary = {
        "run_dir": str(run_dir),
        "method": str(method or getattr(cfg, "method", "")),
        "dataset": dataset_name,
        "seed": str(seed),
        "checkpoint": str(ckpt) if ckpt and ckpt.exists() else "",
        "num_steps": steps,
        "num_step_rows": len(step_rows),
        "num_sample_rows": len(sample_rows),
        "mean_margin": mean(r.get("margin") for r in step_rows),
        "mean_delta_margin_vs_frozen": mean(r.get("delta_margin_vs_frozen") for r in step_rows),
        "status": "real_forward",
    }
    return step_rows, sample_rows, summary
