#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


CELLS = (
    ("Ego-V", "verb", "ego"),
    ("Ego-N", "noun", "ego"),
    ("Exo-V", "verb", "exo"),
    ("Exo-N", "noun", "exo"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate corrected FlyGCL anticipation runs")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.input_root).resolve()
    payloads = {}
    for seed in args.seeds:
        path = root / f"seed_{seed}" / "final_results.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("evaluation_protocol") != "global_macro_top5_seen_sessions":
            raise ValueError(f"Incompatible evaluation protocol in {path}")
        payloads[seed] = payload

    rows = []
    summary = {
        "status": "complete",
        "evaluation_protocol": "global_macro_top5_seen_sessions",
        "seeds": args.seeds,
        "metrics": {},
    }
    for metric in ("A_last", "A_auc"):
        row = {"Metric": metric}
        values_by_cell = {}
        for name, target, view in CELLS:
            values = [payloads[seed]["results"][target][view][metric] for seed in args.seeds]
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            row[name] = f"{mean:.4f}±{std:.4f}"
            values_by_cell[name] = {"mean": mean, "std": std, "values": values}
        seed_averages = [
            statistics.fmean(
                payloads[seed]["results"][target][view][metric]
                for _, target, view in CELLS
            )
            for seed in args.seeds
        ]
        avg_mean = statistics.fmean(seed_averages)
        avg_std = statistics.stdev(seed_averages) if len(seed_averages) > 1 else 0.0
        row["Avg"] = f"{avg_mean:.4f}±{avg_std:.4f}"
        rows.append(row)
        summary["metrics"][metric] = {
            "cells": values_by_cell,
            "Avg": {"mean": avg_mean, "std": avg_std, "values": seed_averages},
        }

    columns = ["Metric", "Ego-V", "Ego-N", "Exo-V", "Exo-N", "Avg"]
    with (root / "table_s8_flygcl_mean_std.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (root / "aggregate_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
