#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flygcl.common.metrics import summarize_accuracy_matrix
from flygcl.skill.utilities import (
    build_graph_margins,
    load_prediction_rows,
    normalized,
    record_metrics,
    snapshot_records,
)


def prediction_path(root: Path, session: int, task: int) -> Path:
    return root / f"task_{session:02d}/predictions/eval_task_{task:02d}.jsonl"


def evaluate(records, counts, alpha: float, tau: float):
    matrix = []
    for session in range(1, 5):
        row = []
        for record in records:
            if record["session"] != session:
                continue
            base = normalized(record["base"])
            auxiliary = normalized(record["auxiliary"])
            gate = np.exp(-np.abs(base) / float(tau))
            margin = base + float(alpha) * gate * auxiliary
            row.append(float((margin > 0).mean()))
        matrix.append(row)
    return matrix, summarize_accuracy_matrix(matrix, counts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the fixed FlyGCL TL fusion")
    parser.add_argument("--expert-run", required=True)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decay", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.56)
    parser.add_argument("--tau", type=float, default=2.5)
    parser.add_argument("--method-name", default="FlyGCL safe cross-view residual")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError(f"Output must remain inside {ROOT}: {output}")
    output.mkdir(parents=True, exist_ok=True)
    graph = build_graph_margins(
        ROOT / "configs/skill_tl.json",
        neighbors=20,
        temperature=0.08,
    )
    records, counts = snapshot_records(Path(args.base_run), graph, args.decay)
    expert_run = Path(args.expert_run)
    for record in records:
        rows = load_prediction_rows(
            prediction_path(expert_run, record["session"], record["task"])
        )
        if set(rows) != set(record["ids"]):
            raise RuntimeError(
                f"FlyGCL sample mismatch at s{record['session']} t{record['task']}"
            )
        record["auxiliary"] = np.asarray([rows[item]["margin"] for item in record["ids"]])
    base_matrix, base_metrics = record_metrics(records, counts)
    matrix, metrics = evaluate(records, counts, args.alpha, args.tau)
    constraints = {
        "A_last_not_lower": metrics["final_macro_accuracy"] >= base_metrics["final_macro_accuracy"],
        "F_T_not_higher": metrics["average_forgetting"] <= base_metrics["average_forgetting"],
        "A_auc_higher": metrics["cl_auc"] > base_metrics["cl_auc"],
    }
    selected = {
        "alpha": args.alpha,
        "tau": args.tau,
        "accuracy_matrix": matrix,
        "metrics": metrics,
        "constraints": constraints,
        "accepted": all(constraints.values()),
    }
    payload = {
        "status": "complete",
        "method": f"{args.method_name} over frozen snapshot anchor",
        "seed": args.seed,
        "mode": "fixed",
        "anchor": {"accuracy_matrix": base_matrix, "metrics": base_metrics},
        "accepted": selected["accepted"],
        "selected": selected,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for variant, entry in (("snapshot_anchor", {"metrics": base_metrics}), ("flygcl", selected)):
        if entry is None:
            continue
        metrics = entry["metrics"]
        rows.append(
            {
                "Variant": variant,
                "A_last_percent": 100 * metrics["final_macro_accuracy"],
                "F_T_percent": 100 * metrics["average_forgetting"],
                "A_auc_percent": 100 * metrics["cl_auc"],
            }
        )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))
    print(f"[done] {output / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
