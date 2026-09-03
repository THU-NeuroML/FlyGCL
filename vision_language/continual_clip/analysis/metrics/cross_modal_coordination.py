from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

from .io_utils import write_csv, write_json, write_md
from .metric_utils import run_metadata


STEP_FIELDS = [
    "step", "method", "seed", "dataset", "group_name", "reference", "class_set_mode",
    "vision_drift_center", "text_drift_center", "modality_drift_depth_gap",
    "vision_peak_drift_layer", "text_peak_drift_layer", "modality_peak_layer_gap",
    "text_boundary_drift", "cross_modal_drift_alignment",
    "margin_delta_visual_contribution", "margin_delta_text_boundary_contribution",
    "margin_delta_interaction_contribution", "status",
]
SAMPLE_FIELDS = [
    "step", "method", "seed", "dataset", "sample_id", "group_name", "reference",
    "text_boundary_drift", "cross_modal_drift_alignment",
    "margin_delta_visual_contribution", "margin_delta_text_boundary_contribution",
    "margin_delta_interaction_contribution", "status",
]


def collect(run_dir: Path, output_dir: Path, args) -> Dict[str, object]:
    meta = run_metadata(run_dir, args.method, args.dataset, str(args.seed), args.config)
    row = {
        **meta,
        "step": 0,
        "group_name": "all",
        "reference": getattr(args, "reference", "initial"),
        "class_set_mode": getattr(args, "class_set_mode", "both"),
        "vision_drift_center": math.nan,
        "text_drift_center": math.nan,
        "modality_drift_depth_gap": math.nan,
        "vision_peak_drift_layer": math.nan,
        "text_peak_drift_layer": math.nan,
        "modality_peak_layer_gap": math.nan,
        "text_boundary_drift": math.nan,
        "cross_modal_drift_alignment": math.nan,
        "margin_delta_visual_contribution": math.nan,
        "margin_delta_text_boundary_contribution": math.nan,
        "margin_delta_interaction_contribution": math.nan,
        "status": "pending_joint_vision_text_feature_and_boundary_loader",
    }
    summary = {
        **meta,
        "num_step_rows": 1,
        "num_sample_rows": 0,
        "status": "pending",
        "interpretation": "CLIP-specific coordination signal; not a claim that schema must be cross-modal.",
    }
    write_csv(output_dir / "cross_modal_coordination_step_metrics.csv", [row], STEP_FIELDS)
    write_csv(output_dir / "cross_modal_coordination_sample_metrics.csv", [], SAMPLE_FIELDS)
    write_json(output_dir / "cross_modal_coordination_summary.json", summary)
    write_md(
        output_dir / "cross_modal_coordination_report.md",
        "Cross-modal Coordination Metrics",
        {
            "Summary": summary,
            "Formula": "delta_margin ~= delta_image dot boundary_ref + image_ref dot delta_boundary + delta_image dot delta_boundary.",
            "Pending": "Requires paired current/reference image features, text prototypes, hard-negative boundaries, and optional layer-depth drift summaries.",
        },
    )
    return summary

