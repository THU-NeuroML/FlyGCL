from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

from .io_utils import mean, read_json, write_csv, write_json, write_md
from .metric_utils import load_gcl_metric_rows, run_metadata


STEP_FIELDS = [
    "step", "method", "seed", "dataset", "run_dir", "session_id", "group_name",
    "old_acc", "new_acc", "future_acc", "seen_acc", "all_acc",
    "avg_acc", "last_acc", "AccAUC", "AccFin", "forgetting", "BWT",
]
CLASS_FIELDS = ["step", "method", "seed", "dataset", "run_dir", "class_id", "per_class_acc", "per_class_acc_drop"]


def collect(run_dir: Path, output_dir: Path, args) -> Dict[str, object]:
    meta = run_metadata(run_dir, args.method, args.dataset, str(args.seed), args.config)
    metric_rows = load_gcl_metric_rows(run_dir)
    step_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []
    for row in metric_rows:
        if row.get("type") != "session_end":
            continue
        sid = int(row.get("session_id", row.get("session", len(step_rows) + 1)))
        step_rows.append({
            "step": sid - 1,
            "method": meta["method"],
            "seed": meta["seed"],
            "dataset": meta["dataset"],
            "run_dir": meta["run_dir"],
            "session_id": sid,
            "group_name": "all",
            "old_acc": row.get("old_exposed_acc", math.nan),
            "new_acc": row.get("new_exposed_acc", math.nan),
            "future_acc": math.nan,
            "seen_acc": row.get("all_exposed_acc", row.get("acc_primary", math.nan)),
            "all_acc": row.get("acc_primary", row.get("accuracy", math.nan)),
            "avg_acc": row.get("acc_avg", row.get("A_avg_post", math.nan)),
            "last_acc": row.get("acc_fin", row.get("A_last_post", math.nan)),
            "AccAUC": row.get("acc_auc", row.get("A_auc", math.nan)),
            "AccFin": row.get("acc_fin", row.get("A_last_post", math.nan)),
            "forgetting": row.get("forgetting", math.nan),
            "BWT": row.get("bwt", row.get("bwt_post", math.nan)),
        })
    leaderboard = read_json(run_dir / "leaderboard_summary.json")
    summary = {
        **meta,
        "num_steps": len(step_rows),
        "avg_acc": mean(r.get("avg_acc") for r in step_rows),
        "last_acc": step_rows[-1]["last_acc"] if step_rows else leaderboard.get("acc_fin", math.nan),
        "AccAUC": leaderboard.get("acc_auc", mean(r.get("AccAUC") for r in step_rows)),
        "AccFin": leaderboard.get("acc_fin", step_rows[-1]["AccFin"] if step_rows else math.nan),
        "status": "adapter_from_gcl_metrics",
    }
    write_csv(output_dir / "level0_performance_step_metrics.csv", step_rows, STEP_FIELDS)
    write_csv(output_dir / "level0_performance_class_metrics.csv", class_rows, CLASS_FIELDS)
    write_json(output_dir / "level0_performance_summary.json", summary)
    write_md(
        output_dir / "level0_performance_report.md",
        "Level 0 Performance Report",
        {
            "Role": "Level 0 is the explained outcome, not a mechanism metric.",
            "Summary": summary,
            "Grouping": "old/new/seen/all are read from GCL metrics when available; future_acc is NaN unless an upstream evaluator records future-class accuracy.",
        },
    )
    return summary
