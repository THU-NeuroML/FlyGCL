from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

from .io_utils import mean, read_csv, write_csv, write_json, write_md
from .metric_utils import first_existing, run_metadata


STEP_FIELDS = [
    "step", "method", "seed", "dataset", "group_name", "reference", "class_set_mode",
    "feature_direction_drift", "feature_l2_drift",
    "class_prototype_drift", "class_prototype_cos_drift", "RDM_drift",
    "margin_projected_feature_drift",
    "text_prototype_drift", "text_boundary_drift", "text_RDM_drift",
    "functional_link_metric", "status",
]
CLASS_FIELDS = [
    "step", "method", "seed", "dataset", "class_id", "reference",
    "class_prototype_drift", "class_prototype_cos_drift", "text_prototype_drift",
]
SAMPLE_FIELDS = [
    "step", "method", "seed", "dataset", "sample_id", "group_name", "reference",
    "feature_direction_drift", "feature_l2_drift", "margin_projected_feature_drift",
    "hard_negative_class_id",
]


def collect(run_dir: Path, output_dir: Path, args) -> Dict[str, object]:
    meta = run_metadata(run_dir, args.method, args.dataset, str(args.seed), args.config)
    path = first_existing(run_dir, ["feature_drift.csv", "feature_posthoc.csv", "prototype_geometry.csv"])
    raw = read_csv(path) if path else []
    step_rows: List[Dict[str, object]] = []
    for source in raw:
        step_rows.append({
            "step": source.get("step_idx", source.get("step", 0)),
            "method": meta["method"],
            "seed": meta["seed"],
            "dataset": meta["dataset"],
            "group_name": source.get("group_name", "all"),
            "reference": getattr(args, "reference", "initial"),
            "class_set_mode": getattr(args, "class_set_mode", "both"),
            "feature_direction_drift": source.get("feature_direction_drift", source.get("mean_feature_drift", source.get("feature_cos_drift", math.nan))),
            "feature_l2_drift": source.get("feature_l2_drift", math.nan),
            "class_prototype_drift": source.get("class_prototype_drift", source.get("prototype_drift", math.nan)),
            "class_prototype_cos_drift": source.get("class_prototype_cos_drift", math.nan),
            "RDM_drift": source.get("RDM_drift", source.get("rdm_drift", math.nan)),
            "margin_projected_feature_drift": source.get("margin_projected_feature_drift", math.nan),
            "text_prototype_drift": source.get("text_prototype_drift", math.nan),
            "text_boundary_drift": source.get("text_boundary_drift", math.nan),
            "text_RDM_drift": source.get("text_RDM_drift", math.nan),
            "functional_link_metric": "old_margin/hard_negative/prototype_relation",
            "status": "adapter_from_existing_records",
        })
    if not step_rows:
        step_rows.append({
            **meta,
            "step": 0,
            "group_name": "all",
            "reference": getattr(args, "reference", "initial"),
            "class_set_mode": getattr(args, "class_set_mode", "both"),
            "feature_direction_drift": math.nan,
            "feature_l2_drift": math.nan,
            "class_prototype_drift": math.nan,
            "class_prototype_cos_drift": math.nan,
            "RDM_drift": math.nan,
            "margin_projected_feature_drift": math.nan,
            "text_prototype_drift": math.nan,
            "text_boundary_drift": math.nan,
            "text_RDM_drift": math.nan,
            "functional_link_metric": "old_margin/hard_negative/prototype_relation",
            "status": "pending_checkpoint_feature_loader",
        })
    class_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    summary = {
        **meta,
        "num_step_rows": len(step_rows),
        "mean_feature_direction_drift": mean(r.get("feature_direction_drift") for r in step_rows),
        "mean_margin_projected_feature_drift": mean(r.get("margin_projected_feature_drift") for r in step_rows),
        "reference": args.reference,
        "status": "implemented_if_upstream_records_exist_else_pending",
    }
    write_csv(output_dir / "level2_feature_step_metrics.csv", step_rows, STEP_FIELDS)
    write_csv(output_dir / "level2_feature_class_metrics.csv", class_rows, CLASS_FIELDS)
    write_csv(output_dir / "level2_feature_sample_metrics.csv", sample_rows if args.save_sample_metrics else [], SAMPLE_FIELDS)
    write_json(output_dir / "level2_feature_summary.json", summary)
    write_md(
        output_dir / "level2_feature_report.md",
        "Level 2 Representation / Relation Drift Report",
        {
            "Summary": summary,
            "Interpretation": "Drift magnitude is observational. Functional interpretation requires linking drift to margin_projected_feature_drift, hard-negative confusion, old margin drop, or RDM/prototype relation changes.",
            "Pending": "Full recomputation requires checkpoint feature loaders for current/ref image features and text boundaries. Missing values are NaN, not fabricated.",
        },
    )
    return summary
