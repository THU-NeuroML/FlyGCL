"""Formal class-exclusive prototype-group continual experiments."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .audit import atomic_json, canonical_sha256, runtime_identity, source_identity
from .config import CONDITIONS, Config, PROJECT_ROOT
from .data import load_seed
from .experiment import Training, train
from .quick_protocol import balanced_prototype_groups, subset_dataset

LEARNING_RATES = (3e-4, 1e-3, 2e-3, 3e-3, 1e-2)
SEEDS = tuple(range(5))
TASKS = tuple((learning_rate, seed, condition) for learning_rate in LEARNING_RATES for seed in SEEDS for condition in CONDITIONS)


def output_path(root: Path, learning_rate: float, seed: int, condition: str) -> Path:
    return root / "runs" / f"lr{learning_rate:.8g}" / f"seed{seed}_{condition}.json"


def execute(task_index: int, device: str, root: Path) -> Path:
    learning_rate, seed, condition = TASKS[task_index]
    output = output_path(root, learning_rate, seed, condition)
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("identity", {}).get("task_index") != task_index:
            raise RuntimeError(f"invalid existing result: {output}")
        print(f"SKIP {task_index:02d} {output}", flush=True)
        return output
    base = Config()
    dataset = load_seed(seed, base)
    class_groups = balanced_prototype_groups(np.asarray(dataset["prototypes"]), base.n_regions, seed)
    train_regions = class_groups[np.asarray(dataset["train_labels"], dtype=np.int64)]
    test_regions = class_groups[np.asarray(dataset["test_labels"], dtype=np.int64)]
    formal, cfg = subset_dataset(dataset, train_regions, test_regions, base, seed, 10_000, 2_000)
    cfg = replace(cfg, evaluation_points_per_region=5, result_root=root)
    identity = {
        "task_index": task_index,
        "learning_rate": learning_rate,
        "seed": seed,
        "condition": condition,
        "train_per_stage": 10_000,
        "test_per_stage": 2_000,
        "class_exclusive": True,
        "classes_per_stage": 20,
    }
    configuration = {
        "identity": identity,
        "class_groups": class_groups.tolist(),
        "protocol": cfg.to_dict(),
        "source": source_identity(PROJECT_ROOT),
        "runtime": runtime_identity(device),
        "data_fingerprint": dataset["metadata"]["fingerprint"],
    }
    started = time.time()
    payload = train(formal, condition, Training(learning_rate, device), seed, cfg)
    payload.update({
        "status": "complete",
        "identity": identity,
        "class_groups": class_groups.tolist(),
        "classes_per_stage": np.bincount(class_groups, minlength=base.n_regions).tolist(),
        "natural_stage_counts": {
            "train": np.bincount(train_regions, minlength=base.n_regions).tolist(),
            "test": np.bincount(test_regions, minlength=base.n_regions).tolist(),
        },
        "fingerprint": {"algorithm": "sha256-canonical-json", "sha256": canonical_sha256(configuration), "configuration": configuration},
        "runtime": {**configuration["runtime"], "seconds": time.time() - started},
    })
    atomic_json(output, payload)
    print(f"DONE {task_index:02d} {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, choices=range(len(TASKS)))
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--result-root", type=Path, default=PROJECT_ROOT / "results" / "disjoint_baseline")
    args = parser.parse_args()
    if args.task_index is not None:
        indices = (args.task_index,)
    else:
        if args.worker is None or args.workers is None or not 0 <= args.worker < args.workers:
            parser.error("provide --task-index or valid --worker/--workers")
        if not 0 <= args.start <= len(TASKS):
            parser.error("--start is outside the task range")
        indices = range(args.start + args.worker, len(TASKS), args.workers)
    for task_index in indices:
        execute(task_index, args.device, args.result_root.resolve())


if __name__ == "__main__":
    main()
