"""Fast screen for class-exclusive prototype-group continual streams."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .audit import atomic_json
from .config import Config
from .data import load_seed
from .experiment import Training, train
from .model import FlyModelV4


def balanced_prototype_groups(prototypes: np.ndarray, n_groups: int, seed: int) -> np.ndarray:
    if len(prototypes) % n_groups:
        raise ValueError("prototype count must be divisible by group count")
    capacity = len(prototypes) // n_groups
    rng = np.random.default_rng(np.random.SeedSequence([2026, 8, 23, 410_000, seed]))
    first = int(np.argmax(np.sum((prototypes - prototypes.mean(0)) ** 2, axis=1)))
    chosen = [first]
    nearest = np.full(len(prototypes), np.inf)
    while len(chosen) < n_groups:
        nearest = np.minimum(nearest, np.sum((prototypes - prototypes[chosen[-1]]) ** 2, axis=1))
        nearest[chosen] = -1
        chosen.append(int(np.argmax(nearest)))
    centers = prototypes[chosen].copy()
    assignment = np.zeros(len(prototypes), dtype=np.uint8)
    for _ in range(30):
        costs = np.sum((prototypes[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        slots = np.repeat(np.arange(n_groups), capacity)
        rows, columns = linear_sum_assignment(costs[:, slots])
        updated = slots[columns].astype(np.uint8)
        new_centers = np.stack([prototypes[updated == group].mean(0) for group in range(n_groups)])
        assignment = updated
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    if not np.array_equal(np.bincount(assignment, minlength=n_groups), np.full(n_groups, capacity)):
        raise RuntimeError("balanced grouping failed")
    return assignment


def encode_subset(model: FlyModelV4, samples: np.ndarray, indices: np.ndarray, device: str, batch_size: int = 2048) -> np.ndarray:
    encoded = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            x = torch.from_numpy(np.array(samples[indices[start:start + batch_size]], dtype=np.float32, copy=True)).to(device)
            encoded.append(model.encode(x).cpu().numpy())
    return np.concatenate(encoded)


def routing_screen(dataset: dict[str, Any], regions: np.ndarray, cfg: Config, seed: int, device: str, per_group: int) -> dict[str, Any]:
    rng = np.random.default_rng(np.random.SeedSequence([2026, 8, 23, 420_000, seed]))
    train_indices = np.concatenate([rng.choice(np.flatnonzero(regions == group), per_group, replace=False) for group in range(cfg.n_regions)])
    train_targets = regions[train_indices]
    test_regions = np.asarray(dataset["test_regions_quick"])
    test_indices = np.concatenate([rng.choice(np.flatnonzero(test_regions == group), per_group, replace=False) for group in range(cfg.n_regions)])
    test_targets = test_regions[test_indices]
    model = FlyModelV4(seed, 1, dataset["orn_pn"], dataset["pn_kc"], cfg, device)
    train_kc = encode_subset(model, dataset["train_samples"], train_indices, device)
    test_kc = encode_subset(model, dataset["test_samples"], test_indices, device)
    prototypes = np.stack([train_kc[train_targets == group].mean(0) for group in range(cfg.n_regions)])
    train_norm = prototypes / np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    test_norm = test_kc / np.maximum(np.linalg.norm(test_kc, axis=1, keepdims=True), 1e-12)
    predicted = np.argmax(test_norm @ train_norm.T, axis=1)
    matrix = np.zeros((cfg.n_regions, cfg.n_regions), dtype=np.int64)
    np.add.at(matrix, (predicted, test_targets), 1)
    return {"accuracy": float(np.mean(predicted == test_targets)), "confusion_predicted_by_true": matrix.tolist(), "samples_per_group": per_group}


def subset_dataset(dataset: dict[str, Any], train_regions: np.ndarray, test_regions: np.ndarray, cfg: Config, seed: int, train_per_group: int, test_per_group: int) -> tuple[dict[str, Any], Config]:
    rng = np.random.default_rng(np.random.SeedSequence([2026, 8, 23, 430_000, seed]))
    train_idx = np.concatenate([rng.choice(np.flatnonzero(train_regions == group), min(train_per_group, int(np.sum(train_regions == group))), replace=False) for group in range(cfg.n_regions)])
    test_idx = np.concatenate([rng.choice(np.flatnonzero(test_regions == group), min(test_per_group, int(np.sum(test_regions == group))), replace=False) for group in range(cfg.n_regions)])
    mini = dict(dataset)
    mini.update({
        "train_samples": np.asarray(dataset["train_samples"][train_idx]),
        "train_labels": np.asarray(dataset["train_labels"][train_idx]),
        "train_regions": train_regions[train_idx],
        "test_samples": np.asarray(dataset["test_samples"][test_idx]),
        "test_labels": np.asarray(dataset["test_labels"][test_idx]),
        "test_regions": test_regions[test_idx],
        "stage_order": np.arange(cfg.n_regions, dtype=np.uint8),
        "metadata": {**dataset["metadata"], "counts": {"train": np.bincount(train_regions[train_idx], minlength=cfg.n_regions).tolist(), "test": np.bincount(test_regions[test_idx], minlength=cfg.n_regions).tolist()}},
    })
    mini_cfg = replace(cfg, n_train=len(train_idx), n_test=len(test_idx), evaluation_points_per_region=1)
    return mini, mini_cfg


def run(seed: int, device: str, output: Path, screen_per_group: int, train_per_group: int, test_per_group: int) -> dict[str, Any]:
    cfg = Config()
    dataset = load_seed(seed, cfg)
    class_groups = balanced_prototype_groups(np.asarray(dataset["prototypes"]), cfg.n_regions, seed)
    train_regions = class_groups[np.asarray(dataset["train_labels"], dtype=np.int64)]
    test_regions = class_groups[np.asarray(dataset["test_labels"], dtype=np.int64)]
    dataset["test_regions_quick"] = test_regions
    screen = routing_screen(dataset, train_regions, cfg, seed, device, screen_per_group)
    mini, mini_cfg = subset_dataset(dataset, train_regions, test_regions, cfg, seed, train_per_group, test_per_group)
    results = {}
    for condition in ("shared", "online_routed_5", "oracle_routed_5"):
        result = train(mini, condition, Training(3e-4, device), seed, mini_cfg)
        results[condition] = {
            "summary": result["summary"],
            "stage_end_region_accuracy": [result["evaluation"]["records"][stage + 1]["region_accuracy"] for stage in range(cfg.n_regions)],
            "final_routing_accuracy": result["evaluation"]["records"][-1]["routing"]["route_accuracy"],
        }
    payload = {
        "schema_version": 1,
        "experiment": "olfactory_quick_class_exclusive_screen",
        "seed": seed,
        "core_definition": "each class belongs to exactly one of five spatially clustered prototype groups",
        "class_groups": class_groups.tolist(),
        "classes_per_stage": np.bincount(class_groups, minlength=cfg.n_regions).tolist(),
        "natural_train_stage_counts": np.bincount(train_regions, minlength=cfg.n_regions).tolist(),
        "natural_test_stage_counts": np.bincount(test_regions, minlength=cfg.n_regions).tolist(),
        "routing_screen": screen,
        "mini_protocol": {"train_per_stage": train_per_group, "test_per_stage": test_per_group, "learning_rate": 3e-4, "evaluation_points_per_stage": 1},
        "results": results,
    }
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--screen-per-group", type=int, default=1000)
    parser.add_argument("--train-per-group", type=int, default=10000)
    parser.add_argument("--test-per-group", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Config().result_root / "quick_protocol" / "seed0.json")
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.device, args.output.resolve(), args.screen_per_group, args.train_per_group, args.test_per_group), indent=2))


if __name__ == "__main__":
    main()
