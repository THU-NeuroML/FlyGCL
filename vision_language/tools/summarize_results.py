#!/usr/bin/env python3
"""Summarize FlyGCL leaderboard JSON files using the historical conventions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


FIELDS = ("dataset", "seed", "A_last", "A_auc", "F", "BWT", "periodic_A_auc", "source")


def find_summaries(inputs: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found.extend(path.rglob("leaderboard_summary.json"))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(raw)
    # Preserve repository-relative provenance when relative inputs are used;
    # do not leak the packager's absolute filesystem path into release CSVs.
    return sorted(set(found))


def load_row(path: Path) -> dict | None:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status", "success") != "success":
        return None
    required = ("dataset", "seed", "acc_fin", "acc_avg", "forgetting", "bwt")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    return {
        "dataset": str(payload["dataset"]),
        "seed": int(payload["seed"]),
        "A_last": float(payload["acc_fin"]),
        # The pipeline stores the mean of session-end accuracies as acc_avg;
        # periodic evaluation is retained separately below.
        "A_auc": float(payload["acc_avg"]),
        "F": float(payload["forgetting"]),
        "BWT": float(payload["bwt"]),
        "periodic_A_auc": float(payload.get("acc_auc", 0.0)),
        "source": str(path),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["dataset"], []).append(row)
    output = []
    for dataset, group in sorted(groups.items()):
        item = {"dataset": dataset, "count": len(group), "std_definition": "population"}
        for metric in ("A_last", "A_auc", "F", "BWT", "periodic_A_auc"):
            values = [float(row[metric]) for row in group]
            item[metric] = {
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
                "formatted": f"{statistics.mean(values):.2f} ± {statistics.pstdev(values):.2f}",
            }
        output.append(item)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="leaderboard_summary.json files or directories")
    parser.add_argument("--csv", dest="csv_path", help="write per-seed CSV")
    parser.add_argument("--json", dest="json_path", help="write per-seed and aggregate JSON")
    args = parser.parse_args()

    rows = [row for path in find_summaries(args.inputs) if (row := load_row(path)) is not None]
    if not rows:
        raise SystemExit("no successful leaderboard summaries found")
    keys = [(row["dataset"], row["seed"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate dataset/seed summaries found; pass a narrower input directory")

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    summary = aggregate(rows)
    for item in summary:
        print(f"\n{item['dataset']} (n={item['count']}, population std)")
        print("  " + " | ".join(f"{key}={item[key]['formatted']}" for key in ("A_last", "A_auc", "F", "BWT")))

    if args.csv_path:
        out = Path(args.csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            file_writer = csv.DictWriter(handle, fieldnames=FIELDS)
            file_writer.writeheader()
            file_writer.writerows(rows)
    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump({"per_seed": rows, "aggregate": summary}, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
