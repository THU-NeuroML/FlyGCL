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
from flygcl.skill.utilities import build_graph_margins, load_prediction_rows, normalized


FIXED_PARAMETERS = {
    "specialist_weight": 0.45,
    "global_weight": 0.35,
    "graph_weight": 0.05,
    "consensus_weight": 0.10,
    "decay": 0.0,
    "confidence_floor": 0.50,
}


def prediction_path(root: Path, session: int, task: int) -> Path:
    return root / f"task_{session:02d}/predictions/eval_task_{task:02d}.jsonl"


def load_records(run: Path, graph: dict):
    records = []
    final_counts = []
    for session in range(1, 5):
        counts = []
        for task in range(1, session + 1):
            current = load_prediction_rows(prediction_path(run, session, task))
            ids = sorted(current)
            histories = [
                load_prediction_rows(prediction_path(run, snapshot, task))
                for snapshot in range(task, session + 1)
            ]
            if any(set(mapping) != set(ids) for mapping in histories):
                raise RuntimeError(f"Temporal sample mismatch at session={session}, task={task}")
            records.append(
                {
                    "session": session,
                    "task": task,
                    "ids": ids,
                    "specialist": np.asarray(
                        [current[item]["specialist_margin"] for item in ids]
                    ),
                    "consensus": np.asarray(
                        [current[item]["expert_consensus_margin"] for item in ids]
                    ),
                    "confidence": np.asarray(
                        [current[item]["route_confidence"] for item in ids]
                    ),
                    "global_history": np.stack(
                        [
                            np.asarray([mapping[item]["global_margin"] for item in ids])
                            for mapping in histories
                        ]
                    ),
                    "graph": np.asarray([graph[task - 1][item] for item in ids]),
                }
            )
            counts.append(len(ids))
        if session == 4:
            final_counts = counts
    return records, final_counts


def temporal_global(history: np.ndarray, decay: float) -> np.ndarray:
    age = np.arange(history.shape[0] - 1, -1, -1, dtype=np.float64)
    weights = np.exp(-float(decay) * age)
    return np.average(history, axis=0, weights=weights)


def evaluate(records, counts, candidate):
    matrix = []
    for session in range(1, 5):
        row = []
        for record in records:
            if record["session"] != session:
                continue
            specialist = normalized(record["specialist"])
            consensus = normalized(record["consensus"])
            global_margin = normalized(
                temporal_global(record["global_history"], candidate["decay"])
            )
            graph = normalized(record["graph"])
            route_weight = candidate["confidence_floor"] + (
                1.0 - candidate["confidence_floor"]
            ) * record["confidence"]
            routed_specialist = route_weight * specialist + (1.0 - route_weight) * global_margin
            margin = (
                candidate["specialist_weight"] * routed_specialist
                + candidate["global_weight"] * global_margin
                + candidate["graph_weight"] * graph
                + candidate["consensus_weight"] * consensus
            )
            row.append(float((margin > 0).mean()))
        matrix.append(row)
    return matrix, summarize_accuracy_matrix(matrix, counts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the fixed FlyGCL Ego-only fusion")
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError(f"Output must remain inside {ROOT}: {output}")
    output.mkdir(parents=True, exist_ok=True)
    graph = build_graph_margins(Path(args.config), neighbors=24, temperature=0.07)
    records, counts = load_records(Path(args.run), graph)

    matrix, metrics = evaluate(records, counts, FIXED_PARAMETERS)
    selected = {**FIXED_PARAMETERS, "accuracy_matrix": matrix, "metrics": metrics}
    payload = {
        "status": "complete",
        "method": "FlyGCL enhanced Ego-only temporal MoE",
        "seed": args.seed,
        "mode": "fixed",
        "fixed_parameters": dict(FIXED_PARAMETERS),
        "selected": selected,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = [
        {
            "Variant": "enhanced_ego_temporal_moe",
            "A_last_percent": 100 * selected["metrics"]["final_macro_accuracy"],
            "F_T_percent": 100 * selected["metrics"]["average_forgetting"],
            "A_auc_percent": 100 * selected["metrics"]["cl_auc"],
        },
    ]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"[done] {output / 'metrics.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
