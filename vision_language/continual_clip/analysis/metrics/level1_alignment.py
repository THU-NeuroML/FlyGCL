from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

from .io_utils import mean, read_csv, write_csv, write_json, write_md
from .level1_recompute import LEVEL1_SAMPLE_FIELDS, LEVEL1_STEP_FIELDS, modes_from_arg, recompute_frozen_reference, recompute_method_alignment
from .metric_utils import first_existing, prefixed_group_rows, run_metadata


STEP_FIELDS = LEVEL1_STEP_FIELDS
SAMPLE_FIELDS = LEVEL1_SAMPLE_FIELDS


def _alignment_rows(run_dir: Path) -> List[Dict[str, str]]:
    path = first_existing(run_dir, ["clip_alignment.csv", "alignment_posthoc.csv", "clip_alignment_all_groups_long.csv"])
    return read_csv(path) if path else []


def collect(run_dir: Path, output_dir: Path, args) -> Dict[str, object]:
    meta = run_metadata(run_dir, args.method, args.dataset, str(args.seed), args.config)
    if not bool(getattr(args, "dry_run", False)):
        config_path = Path(args.config) if args.config else (run_dir / "config.yaml")
        groups = list(getattr(args, "groups", ["old", "new", "future"]))
        modes = modes_from_arg(getattr(args, "class_set_mode", "both"))
        max_steps = int(getattr(args, "max_steps", 0) or 0) or None
        frozen_rows, _, frozen_summary = recompute_frozen_reference(
            config_path=config_path,
            dataset_name=str(meta.get("dataset") or args.dataset or "unknown"),
            split="test",
            class_set_modes=modes,
            groups=groups,
            batch_size=int(getattr(args, "batch_size", 128)),
            device=str(getattr(args, "device", "cuda")),
            max_samples_per_group=int(getattr(args, "max_samples_per_group", -1)),
            max_steps=max_steps,
        )
        step_rows, sample_rows, summary = recompute_method_alignment(
            config_path=config_path,
            run_dir=run_dir,
            method=str(meta.get("method") or args.method or ""),
            dataset_name=str(meta.get("dataset") or args.dataset or "unknown"),
            seed=str(meta.get("seed") or args.seed or ""),
            class_set_modes=modes,
            groups=groups,
            batch_size=int(getattr(args, "batch_size", 128)),
            device=str(getattr(args, "device", "cuda")),
            max_samples_per_group=int(getattr(args, "max_samples_per_group", -1)),
            frozen_step_rows=frozen_rows,
            max_steps=max_steps,
        )
        summary["frozen_reference_status"] = frozen_summary.get("status", "unknown")
        write_csv(output_dir / "level1_alignment_step_metrics.csv", step_rows, LEVEL1_STEP_FIELDS)
        write_csv(
            output_dir / "level1_alignment_sample_metrics.csv",
            sample_rows if bool(getattr(args, "save_sample_metrics", False)) else sample_rows,
            LEVEL1_SAMPLE_FIELDS,
        )
        write_json(output_dir / "level1_alignment_summary.json", summary)
        write_md(
            output_dir / "level1_alignment_report.md",
            "Level 1 Alignment Report",
            {
                "Summary": summary,
                "Forward Mode": "real checkpoint/frozen recomputation",
                "Notes": "delta_*_vs_frozen and extra_*_margin_drop are computed by matching method rows to frozen reference rows.",
            },
        )
        return summary

    raw_rows = _alignment_rows(run_dir)
    groups = args.groups
    group_rows = prefixed_group_rows(raw_rows, groups)
    step_rows: List[Dict[str, object]] = []
    if not raw_rows:
        step_rows.append({**meta, "step": 0, "group_name": "all"})
    for group in groups:
        source = group_rows.get(group, {})
        row = {
            "step": source.get("step_idx", source.get("step", 0)),
            "method": meta["method"],
            "seed": meta["seed"],
            "dataset": meta["dataset"],
            "group_name": group,
            "pos_cos": source.get("pos_cos_mean", source.get("mean_pos_cos", source.get(f"{group}_pos_cos", math.nan))),
            "max_neg_cos": source.get("max_neg_cos_mean", source.get("mean_max_neg_cos", source.get(f"{group}_max_neg_cos", math.nan))),
            "margin": source.get("margin_mean", source.get("mean_margin", source.get(f"{group}_margin", math.nan))),
            "mean_neg_cos": source.get("mean_neg_cos", math.nan),
            "topk_neg_cos": source.get("topk_neg_cos", math.nan),
            "hard_negative_class_id": source.get("hard_negative_class_id", ""),
            "hard_negative_similarity": source.get("hard_negative_similarity", math.nan),
            "hard_negative_group": source.get("hard_negative_group", ""),
            "old_to_new_hard_negative_ratio": source.get("old_to_new_hard_negative_ratio", math.nan),
            "old_to_future_hard_negative_ratio": source.get("old_to_future_hard_negative_ratio", math.nan),
            "new_to_old_hard_negative_ratio": source.get("new_to_old_hard_negative_ratio", math.nan),
            "delta_pos_cos_vs_initial": source.get("delta_pos_cos_vs_initial", math.nan),
            "delta_margin_vs_initial": source.get("delta_margin_vs_initial", math.nan),
            "delta_margin_vs_frozen": source.get("delta_margin_vs_frozen", math.nan),
            "delta_max_neg_vs_initial": source.get("delta_max_neg_vs_initial", math.nan),
        }
        for g in ("old", "new", "future"):
            gsrc = group_rows.get(g, {})
            row[f"{g}_pos_cos"] = gsrc.get("pos_cos_mean", gsrc.get("mean_pos_cos", math.nan))
            row[f"{g}_margin"] = gsrc.get("margin_mean", gsrc.get("mean_margin", math.nan))
            row[f"{g}_max_neg_cos"] = gsrc.get("max_neg_cos_mean", gsrc.get("mean_max_neg_cos", math.nan))
        step_rows.append(row)
    sample_rows: List[Dict[str, object]] = []
    summary = {**meta, "num_step_rows": len(step_rows), "mean_margin": mean(r.get("margin") for r in step_rows)}
    write_csv(output_dir / "level1_alignment_step_metrics.csv", step_rows, STEP_FIELDS)
    if args.save_sample_metrics:
        write_csv(output_dir / "level1_alignment_sample_metrics.csv", sample_rows, SAMPLE_FIELDS)
    else:
        write_csv(output_dir / "level1_alignment_sample_metrics.csv", [], SAMPLE_FIELDS)
    write_json(output_dir / "level1_alignment_summary.json", summary)
    write_md(output_dir / "level1_alignment_report.md", "Level 1 Alignment Report", {"Summary": summary, "Limitations": "Adapter over existing analysis CSVs; sample-level recomputation requires model features."})
    return summary
