from typing import Dict, Iterable, Mapping, Optional, Set

import torch

from .io_utils import warn_once


def _to_int_set(values: Optional[Iterable]) -> Set[int]:
    if values is None:
        return set()
    if torch.is_tensor(values):
        values = values.detach().cpu().view(-1).tolist()
    return {int(v) for v in values}


def build_old_new_groups(labels, seen_class_ids=None, current_class_ids=None):
    """
    Build boolean masks for the minimal online analysis groups.

    labels are global class ids. If current_class_ids is unavailable, only the
    all group is returned so callers keep the old analysis behavior.
    """
    if labels is None:
        return {}
    labels_t = labels.detach().cpu().long().view(-1) if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.long).view(-1)
    groups = {"all": torch.ones(int(labels_t.numel()), dtype=torch.bool)}
    current = _to_int_set(current_class_ids)
    if not current:
        warn_once("current_class_ids missing; group-wise analysis falls back to group_name=all")
        return groups

    seen = _to_int_set(seen_class_ids)
    if not seen:
        warn_once("seen_class_ids missing; infer seen classes from labels for old/new analysis")
        seen = {int(v) for v in labels_t.tolist()}
    old = set(seen) - set(current)

    current_tensor = torch.tensor(sorted(current), dtype=torch.long)
    old_tensor = torch.tensor(sorted(old), dtype=torch.long)
    groups["new_current"] = torch.isin(labels_t, current_tensor) if current_tensor.numel() > 0 else torch.zeros_like(labels_t, dtype=torch.bool)
    groups["old_all"] = torch.isin(labels_t, old_tensor) if old_tensor.numel() > 0 else torch.zeros_like(labels_t, dtype=torch.bool)
    if not bool(groups["old_all"].any()):
        warn_once("old_all group is empty for this analysis step")
    if not bool(groups["new_current"].any()):
        warn_once("new_current group is empty for this analysis step")
    return groups


def _score_from_history(history, class_id: int):
    if history is None:
        return None
    if isinstance(history, Mapping):
        value = history.get(class_id, history.get(str(class_id), None))
    else:
        try:
            value = history[class_id]
        except Exception:
            value = None
    if isinstance(value, (list, tuple)):
        if len(value) < 2:
            return None
        return float(value[-1]) - float(max(value[:-1]))
    if isinstance(value, Mapping):
        if "forgetting" in value:
            return float(value["forgetting"])
        if "best_acc" in value and "current_acc" in value:
            return float(value["best_acc"]) - float(value["current_acc"])
    return None


def _top_ratio(classes, score_fn, ratio: float, reverse: bool = True) -> Set[int]:
    scored = []
    for cls in classes:
        score = score_fn(cls)
        if score is not None:
            scored.append((int(cls), float(score)))
    if not scored:
        return set()
    scored.sort(key=lambda item: item[1], reverse=reverse)
    count = max(1, int(round(len(scored) * float(ratio))))
    return {cls for cls, _ in scored[:count]}


def build_class_groups(
    step_idx,
    class_order=None,
    current_classes=None,
    seen_classes=None,
    class_acc_history=None,
    class_frequency=None,
    forgotten_top_ratio=0.2,
    preserved_bottom_ratio=0.2,
    major_top_ratio=0.5,
) -> Dict[str, Set[int]]:
    current = _to_int_set(current_classes)
    seen = _to_int_set(seen_classes)
    if not seen and class_order is not None:
        ordered = list(class_order)
        try:
            seen = {int(v) for v in ordered[: int(step_idx) + 1]}
        except Exception:
            seen = _to_int_set(ordered)

    all_classes = set(seen) | set(current)
    old_all = set(seen) - set(current)
    new_current = set(current)

    if class_acc_history is None:
        warn_once("class_acc_history missing; old_forgotten/old_preserved are empty")
        old_forgotten = set()
        old_preserved = set()
    else:
        old_forgotten = _top_ratio(
            old_all,
            lambda cls: _score_from_history(class_acc_history, cls),
            forgotten_top_ratio,
            reverse=True,
        )
        old_preserved = _top_ratio(
            old_all,
            lambda cls: _score_from_history(class_acc_history, cls),
            preserved_bottom_ratio,
            reverse=False,
        )
        if not old_forgotten and old_all:
            warn_once("class_acc_history did not contain usable forgetting scores")

    if class_frequency is None:
        warn_once("class_frequency missing; major/minor are empty")
        major = set()
        minor = set()
    else:
        def freq_score(cls):
            if isinstance(class_frequency, Mapping):
                value = class_frequency.get(cls, class_frequency.get(str(cls), None))
            else:
                try:
                    value = class_frequency[cls]
                except Exception:
                    value = None
            return None if value is None else float(value)

        major = _top_ratio(all_classes, freq_score, major_top_ratio, reverse=True)
        minor = set(all_classes) - set(major) if major else set()
        if not major and all_classes:
            warn_once("class_frequency did not contain usable frequency scores")

    return {
        "all": all_classes,
        "old_all": old_all,
        "new_current": new_current,
        "old_forgotten": old_forgotten,
        "old_preserved": old_preserved,
        "major": major,
        "minor": minor,
    }
