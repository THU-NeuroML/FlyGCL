#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate FlyGCL Ego-exo results")
    parser.add_argument("--head", choices=("rn", "tl"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError(f"Output must remain inside {ROOT}: {output}")
    output.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for seed in args.seeds:
        path = root / f"seed_{seed}" / f"{args.head}_evaluation_fixed" / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("mode") != "fixed_zero_search":
            raise RuntimeError(f"Expected zero-search fixed evaluation: {path}")
        per_seed[str(seed)] = payload["selected"]["metrics"]
    mapping = {
        "A_last": "final_macro_accuracy",
        "F_T": "average_forgetting",
        "A_auc": "cl_auc",
    }
    summary = {}
    for label, key in mapping.items():
        values = np.asarray([100.0 * per_seed[str(seed)][key] for seed in args.seeds])
        summary[label] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "values": [float(value) for value in values],
            "formatted": f"{values.mean():.2f}±{values.std(ddof=0):.2f}",
        }
    payload = {
        "status": "complete",
        "method": f"FlyGCL {args.head.upper()}",
        "seeds": args.seeds,
        "per_seed": per_seed,
        "summary_percent": summary,
    }
    (output / "multiseed_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output / "multiseed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("Head", "Seeds", "A_last", "F_T", "A_auc"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "Head": args.head.upper(),
                "Seeds": "/".join(map(str, args.seeds)),
                "A_last": summary["A_last"]["formatted"],
                "F_T": summary["F_T"]["formatted"],
                "A_auc": summary["A_auc"]["formatted"],
            }
        )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
