import math
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd


CLASSWISE_FIELDS = [
    "method",
    "seed",
    "step_idx",
    "class_id",
    "group_names",
    "acc_c",
    "forgetting_c",
    "pos_cos_c",
    "max_neg_cos_c",
    "margin_c",
    "hard_neg_class_id",
    "hard_neg_is_new",
    "hard_neg_similarity",
    "correct_similarity",
    "feature_drift_c",
    "prototype_drift_c",
    "early_attention_drift_c",
    "mid_attention_drift_c",
    "high_attention_drift_c",
    "early_topk_overlap_c",
    "mid_topk_overlap_c",
    "high_topk_overlap_c",
    "attention_distance_shift_c",
]


def _metric(metrics, class_id: int, *names):
    if metrics is None:
        return math.nan
    entry = metrics.get(class_id, metrics.get(str(class_id), None)) if isinstance(metrics, Mapping) else None
    if entry is None:
        return math.nan
    if not isinstance(entry, Mapping):
        return entry
    for name in names:
        if name in entry:
            return entry[name]
    return math.nan


def _classes_from_groups(group_dict: Optional[Dict[str, Iterable[int]]]):
    if not group_dict:
        return []
    classes = set()
    for values in group_dict.values():
        classes.update(int(v) for v in values)
    return sorted(classes)


def _group_names_for_class(group_dict, class_id: int) -> str:
    if not group_dict:
        return ""
    names = [name for name, values in group_dict.items() if int(class_id) in {int(v) for v in values}]
    return "|".join(sorted(names))


def build_classwise_analysis(
    method_name,
    seed,
    step_idx,
    class_metrics=None,
    alignment_metrics=None,
    feature_metrics=None,
    attention_metrics=None,
    group_dict=None,
):
    rows = []
    for class_id in _classes_from_groups(group_dict):
        rows.append({
            "method": method_name,
            "seed": int(seed) if seed is not None else math.nan,
            "step_idx": int(step_idx) if step_idx is not None else math.nan,
            "class_id": int(class_id),
            "group_names": _group_names_for_class(group_dict, class_id),
            "acc_c": _metric(class_metrics, class_id, "acc", "acc_c"),
            "forgetting_c": _metric(class_metrics, class_id, "forgetting", "forgetting_c"),
            "pos_cos_c": _metric(alignment_metrics, class_id, "pos_cos", "pos_cos_c"),
            "max_neg_cos_c": _metric(alignment_metrics, class_id, "max_neg_cos", "max_neg_cos_c"),
            "margin_c": _metric(alignment_metrics, class_id, "margin", "margin_c"),
            "hard_neg_class_id": _metric(alignment_metrics, class_id, "hard_neg_class_id"),
            "hard_neg_is_new": _metric(alignment_metrics, class_id, "hard_neg_is_new"),
            "hard_neg_similarity": _metric(alignment_metrics, class_id, "hard_neg_similarity"),
            "correct_similarity": _metric(alignment_metrics, class_id, "correct_similarity"),
            "feature_drift_c": _metric(feature_metrics, class_id, "feature_drift", "feature_drift_c"),
            "prototype_drift_c": _metric(feature_metrics, class_id, "prototype_drift", "prototype_drift_c"),
            "early_attention_drift_c": _metric(attention_metrics, class_id, "early_attention_drift"),
            "mid_attention_drift_c": _metric(attention_metrics, class_id, "mid_attention_drift"),
            "high_attention_drift_c": _metric(attention_metrics, class_id, "high_attention_drift"),
            "early_topk_overlap_c": _metric(attention_metrics, class_id, "early_topk_overlap"),
            "mid_topk_overlap_c": _metric(attention_metrics, class_id, "mid_topk_overlap"),
            "high_topk_overlap_c": _metric(attention_metrics, class_id, "high_topk_overlap"),
            "attention_distance_shift_c": _metric(attention_metrics, class_id, "attention_distance_shift"),
        })
    return pd.DataFrame(rows, columns=CLASSWISE_FIELDS)


def empty_classwise_analysis():
    return pd.DataFrame(columns=CLASSWISE_FIELDS)
