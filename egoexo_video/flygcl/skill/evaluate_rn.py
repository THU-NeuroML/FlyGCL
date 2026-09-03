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
from flygcl.skill.records import (
    frozen_ego_records,
    rn_records,
    tl_records,
)
from flygcl.skill.utilities import normalized


FIXED_HYBRID = {"alpha": -0.18, "tau": 0.10, "consensus_weight": 0.20}


def summarize(margins, counts):
    matrix = [
        [float((margins[(session, task)] > 0).mean()) for task in range(1, session + 1)]
        for session in range(1, 5)
    ]
    return matrix, summarize_accuracy_matrix(matrix, counts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed FlyGCL RN/TL hybrid expert")
    parser.add_argument("--ego-run", required=True)
    parser.add_argument("--rn-run", required=True)
    parser.add_argument("--tl-base", required=True)
    parser.add_argument("--tl-expert", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError(f"Output must remain inside {ROOT}: {output}")
    output.mkdir(parents=True, exist_ok=True)

    ego, counts = frozen_ego_records(Path(args.ego_run), Path(args.config))
    tl = tl_records(Path(args.tl_base), Path(args.tl_expert), Path(args.config))
    rn = rn_records(Path(args.rn_run))
    tl_anchor = {}
    hybrid = {}
    for key in ego:
        if ego[key]["ids"] != tl[key]["ids"] or ego[key]["ids"] != rn[key]["ids"]:
            raise RuntimeError(f"Sample mismatch at {key}")
        ego_margin = ego[key]["margin"]
        tl_residual = normalized(tl[key]["margin"]) + 0.30 * normalized(
            tl[key]["consensus"]
        )
        anchor = ego_margin + 0.70 * np.exp(-np.abs(ego_margin) / 5.0) * tl_residual
        rn_residual = normalized(rn[key]["margin"]) + FIXED_HYBRID[
            "consensus_weight"
        ] * normalized(rn[key]["consensus"])
        gate = np.exp(-np.abs(normalized(anchor)) / FIXED_HYBRID["tau"])
        tl_anchor[key] = anchor
        hybrid[key] = anchor + FIXED_HYBRID["alpha"] * gate * rn_residual

    anchor_matrix, anchor_metrics = summarize(tl_anchor, counts)
    matrix, metrics = summarize(hybrid, counts)
    payload = {
        "status": "complete",
        "method": "FlyGCL RN signed correction over frozen TL/Ego expert ensemble",
        "seed": args.seed,
        "mode": "fixed_zero_search",
        "selection_disclosure": (
            "Signed RN correction parameters were selected on seed 42 and are fixed for "
            "all reported runs; no search is performed by this evaluator."
        ),
        "fixed_parameters": FIXED_HYBRID,
        "tl_anchor": {"accuracy_matrix": anchor_matrix, "metrics": anchor_metrics},
        "selected": {
            **FIXED_HYBRID,
            "accuracy_matrix": matrix,
            "metrics": metrics,
        },
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = [
        {
            "Variant": "frozen_tl_anchor",
            "A_last_percent": 100 * anchor_metrics["final_macro_accuracy"],
            "F_T_percent": 100 * anchor_metrics["average_forgetting"],
            "A_auc_percent": 100 * anchor_metrics["cl_auc"],
        },
        {
            "Variant": "rn_hybrid_signed_correction",
            "A_last_percent": 100 * metrics["final_macro_accuracy"],
            "F_T_percent": 100 * metrics["average_forgetting"],
            "A_auc_percent": 100 * metrics["cl_auc"],
        },
    ]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
