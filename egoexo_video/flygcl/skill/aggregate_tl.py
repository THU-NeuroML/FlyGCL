#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = (
    ("A_last", "final_macro_accuracy"),
    ("F_T", "average_forgetting"),
    ("A_auc", "cl_auc"),
)


def summary(entries: dict[int, dict]) -> dict:
    output = {}
    for label, key in METRICS:
        values = [100.0 * entries[seed][key] for seed in sorted(entries)]
        output[label] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate fixed FlyGCL TL multiseed results")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    per_seed = {}
    anchors = {}
    methods = {}
    for seed in args.seeds:
        path = root / f"seed_{seed}/fusion/results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("mode") != "fixed":
            raise RuntimeError(f"Expected fixed result: {path}")
        selected = payload.get("selected")
        if selected is None:
            raise RuntimeError(f"Missing fixed FlyGCL result: {path}")
        anchors[seed] = payload["anchor"]["metrics"]
        methods[seed] = selected["metrics"]
        per_seed[str(seed)] = {
            "accepted": payload["accepted"],
            "constraints": selected["constraints"],
            "anchor": payload["anchor"],
            "flygcl": selected,
        }
    anchor_summary = summary(anchors)
    method_summary = summary(methods)
    result = {
        "status": "complete",
        "method": "FlyGCL fixed cross-view residual",
        "setting": "TL Ego-exo",
        "seeds": args.seeds,
        "fixed_hyperparameters": {"alpha": 0.56, "tau": 2.5, "snapshot_decay": 0.2},
        "selection_protocol": (
            "Alpha/tau selected once on seed 42 and frozen before seeds 43/44. "
            "Seed 43/44 evaluations perform zero search."
        ),
        "accepted_all_seeds": all(item["accepted"] for item in per_seed.values()),
        "anchor_summary": anchor_summary,
        "flygcl_summary": method_summary,
        "per_seed": per_seed,
    }
    (root / "multiseed_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = []
    for seed in args.seeds:
        for variant, metrics in (("snapshot_anchor", anchors[seed]), ("flygcl", methods[seed])):
            rows.append(
                {
                    "Seed": seed,
                    "Variant": variant,
                    "A_last_percent": 100 * metrics["final_macro_accuracy"],
                    "F_T_percent": 100 * metrics["average_forgetting"],
                    "A_auc_percent": 100 * metrics["cl_auc"],
                    "Accepted": per_seed[str(seed)]["accepted"] if variant == "flygcl" else True,
                }
            )
    for variant, values in (("snapshot_anchor_mean", anchor_summary), ("flygcl_mean", method_summary)):
        rows.append(
            {
                "Seed": "mean",
                "Variant": variant,
                "A_last_percent": values["A_last"]["mean"],
                "F_T_percent": values["F_T"]["mean"],
                "A_auc_percent": values["A_auc"]["mean"],
                "Accepted": result["accepted_all_seeds"],
            }
        )
    with (root / "multiseed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))
    print(f"[done] {root / 'multiseed_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
