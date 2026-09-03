from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

from .io_utils import mean, read_csv, write_csv, write_json, write_md
from .metric_utils import first_existing, run_metadata


LH_FIELDS = [
    "step", "method", "seed", "dataset", "group_name", "modality", "layer", "head",
    "anchor_token", "target_token_group", "metric_name", "metric_value",
    "reference", "class_set_mode", "status",
]
STEP_FIELDS = [
    "step", "method", "seed", "dataset", "group_name", "modality", "reference", "class_set_mode",
    "CLS_attention_JS_drift", "CLS_attention_cos_drift", "topk_patch_overlap",
    "attention_entropy", "vision_effective_context_drift",
    "EOT_attention_JS_drift", "EOT_attention_cos_drift", "EOT_to_class_name_attention",
    "EOT_to_template_attention", "EOT_to_prompt_attention", "EOT_attention_entropy",
    "text_effective_context_drift", "functional_consequence",
]
SAMPLE_FIELDS = [
    "step", "method", "seed", "dataset", "sample_id", "group_name", "modality",
    "layer", "head", "anchor_token", "metric_name", "metric_value",
]


def collect(run_dir: Path, output_dir: Path, args) -> Dict[str, object]:
    meta = run_metadata(run_dir, args.method, args.dataset, str(args.seed), args.config)
    path = first_existing(run_dir, ["attention_stats.csv", "attention_stage_summary.csv", "attention_method_comparison.csv"])
    raw = read_csv(path) if path else []
    layers = {int(x) for x in args.layers} if args.layers else None
    layer_head_rows: List[Dict[str, object]] = []
    step_rows: List[Dict[str, object]] = []
    for source in raw:
        layer = source.get("layer", source.get("layer_idx", source.get("visual_layer", "")))
        try:
            layer_i = int(float(layer))
        except Exception:
            layer_i = -1
        if layers is not None and layer_i not in layers:
            continue
        group = source.get("group_name", "all")
        modality = source.get("modality", "vision")
        anchor_token = "EOT" if str(modality) == "text" else "CLS"
        metric_map = {
            "CLS_attention_JS_drift": source.get("CLS_attention_JS_drift", source.get("attention_js_drift", source.get("mean_attention_drift", math.nan))),
            "CLS_attention_cos_drift": source.get("CLS_attention_cos_drift", source.get("attention_cos_drift", math.nan)),
            "topk_patch_overlap": source.get("topk_patch_overlap", source.get("topk_attention_overlap", math.nan)),
            "attention_entropy": source.get("attention_entropy", math.nan),
            "vision_effective_context_drift": source.get("vision_effective_context_drift", source.get("effective_context_drift", math.nan)),
            "EOT_attention_JS_drift": source.get("EOT_attention_JS_drift", math.nan),
            "EOT_attention_cos_drift": source.get("EOT_attention_cos_drift", math.nan),
            "EOT_to_class_name_attention": source.get("EOT_to_class_name_attention", math.nan),
            "EOT_to_template_attention": source.get("EOT_to_template_attention", math.nan),
            "EOT_to_prompt_attention": source.get("EOT_to_prompt_attention", math.nan),
            "EOT_attention_entropy": source.get("EOT_attention_entropy", math.nan),
            "text_effective_context_drift": source.get("text_effective_context_drift", math.nan),
        }
        for metric_name, metric_value in metric_map.items():
            layer_head_rows.append({
                "step": source.get("step_idx", source.get("step", 0)),
                "method": meta["method"],
                "seed": meta["seed"],
                "dataset": meta["dataset"],
                "group_name": group,
                "modality": modality,
                "layer": layer_i if layer_i >= 0 else "",
                "head": source.get("head", source.get("head_idx", "all")),
                "anchor_token": anchor_token,
                "target_token_group": source.get("target_token_group", "patch" if anchor_token == "CLS" else "all"),
                "metric_name": metric_name,
                "metric_value": metric_value,
                "reference": getattr(args, "reference", "initial"),
                "class_set_mode": getattr(args, "class_set_mode", "both"),
                "status": "adapter_from_existing_records",
            })
        step_rows.append({
            "step": source.get("step_idx", 0),
            "method": meta["method"],
            "seed": meta["seed"],
            "dataset": meta["dataset"],
            "group_name": group,
            "modality": modality,
            "reference": getattr(args, "reference", "initial"),
            "class_set_mode": getattr(args, "class_set_mode", "both"),
            "functional_consequence": source.get("functional_consequence", "link_to_margin_or_accuracy_required"),
            **metric_map,
        })
    if not step_rows:
        step_rows.append({
            **meta,
            "step": 0,
            "group_name": "all",
            "modality": "vision",
            "reference": getattr(args, "reference", "initial"),
            "class_set_mode": getattr(args, "class_set_mode", "both"),
            "CLS_attention_JS_drift": math.nan,
            "vision_effective_context_drift": math.nan,
            "functional_consequence": "pending_value_hook_and_metric_linkage",
        })
    sample_rows: List[Dict[str, object]] = []
    summary = {
        **meta,
        "num_layer_head_rows": len(layer_head_rows),
        "mean_CLS_attention_JS_drift": mean(r.get("CLS_attention_JS_drift") for r in step_rows),
        "vision_effective_context_drift_status": "available_if_upstream_value_context_recorded_else_pending",
        "text_effective_context_drift_status": "pending_text_attention_value_hook",
    }
    write_csv(output_dir / "level3_attention_layer_head_metrics.csv", layer_head_rows, LH_FIELDS)
    write_csv(output_dir / "level3_attention_step_metrics.csv", step_rows, STEP_FIELDS)
    write_csv(output_dir / "level3_attention_sample_metrics.csv", sample_rows if args.save_sample_metrics else [], SAMPLE_FIELDS)
    write_json(output_dir / "level3_attention_summary.json", summary)
    write_md(
        output_dir / "level3_attention_report.md",
        "Level 3 Attention / Effective Context Drift Report",
        {
            "Summary": summary,
            "Interpretation": "Raw attention drift is observational. It should only be used with functional consequence metrics such as margin drop, hard-negative confusion, or effective-context drift.",
            "Pending Hooks": "vision/text effective_context_drift requires hooks that capture attention probabilities and value tensors, then compute attention @ value for CLS/EOT anchors.",
        },
    )
    return summary
