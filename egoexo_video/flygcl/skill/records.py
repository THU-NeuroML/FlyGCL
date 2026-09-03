#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flygcl.common.metrics import summarize_accuracy_matrix
from flygcl.skill.evaluate_ego import (
    FIXED_PARAMETERS,
    load_records,
    temporal_global,
)
from flygcl.skill.utilities import (
    build_graph_margins,
    load_prediction_rows,
    normalized,
    snapshot_records,
)


def prediction_path(root: Path, session: int, task: int) -> Path:
    return root / f"task_{session:02d}/predictions/eval_task_{task:02d}.jsonl"


def frozen_ego_records(ego_run: Path, config: Path):
    graph = build_graph_margins(config, neighbors=24, temperature=0.07)
    raw_records, counts = load_records(ego_run, graph)
    output = {}
    cfg = FIXED_PARAMETERS
    for record in raw_records:
        specialist = normalized(record["specialist"])
        consensus = normalized(record["consensus"])
        global_margin = normalized(
            temporal_global(record["global_history"], cfg["decay"])
        )
        graph_margin = normalized(record["graph"])
        route_weight = cfg["confidence_floor"] + (1.0 - cfg["confidence_floor"]) * record[
            "confidence"
        ]
        routed = route_weight * specialist + (1.0 - route_weight) * global_margin
        margin = (
            cfg["specialist_weight"] * routed
            + cfg["global_weight"] * global_margin
            + cfg["graph_weight"] * graph_margin
            + cfg["consensus_weight"] * consensus
        )
        output[(record["session"], record["task"])] = {
            "ids": record["ids"],
            "margin": normalized(margin),
        }
    return output, counts


def tl_records(base_run: Path, expert_run: Path, config: Path):
    graph = build_graph_margins(config, neighbors=20, temperature=0.08)
    records, _ = snapshot_records(base_run, graph, decay=0.2)
    output = {}
    for record in records:
        rows = load_prediction_rows(
            prediction_path(expert_run, record["session"], record["task"])
        )
        auxiliary = np.asarray([rows[item]["margin"] for item in record["ids"]])
        anchor = normalized(record["base"])
        gate = np.exp(-np.abs(anchor) / 2.5)
        margin = anchor + 0.56 * gate * normalized(auxiliary)
        output[(record["session"], record["task"])] = {
            "ids": record["ids"],
            "margin": normalized(margin),
            "consensus": normalized(auxiliary),
        }
    return output


def rn_records(run: Path):
    output = {}
    for session in range(1, 5):
        for task in range(1, session + 1):
            rows = load_prediction_rows(prediction_path(run, session, task))
            ids = sorted(rows)
            output[(session, task)] = {
                "ids": ids,
                "margin": normalized(np.asarray([rows[item]["margin"] for item in ids])),
                "consensus": normalized(
                    np.asarray([rows[item]["expert_consensus_margin"] for item in ids])
                ),
            }
    return output


def metrics(margins, counts):
    matrix = [
        [float((margins[(session, task)] > 0).mean()) for task in range(1, session + 1)]
        for session in range(1, 5)
    ]
    return matrix, summarize_accuracy_matrix(matrix, counts)

