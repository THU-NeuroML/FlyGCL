"""Audit and aggregate class-exclusive formal experiments."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .audit import atomic_json
from .class_exclusive_run import LEARNING_RATES, SEEDS, TASKS, output_path
from .config import CONDITIONS, PROJECT_ROOT
from .experiment import continual_metrics

METRICS = ("seen_anytime_auc", "final_accuracy", "current_adaptation", "old_retention", "average_forgetting", "worst_region_accuracy")


def close(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=1e-8, atol=1e-9))


def validate(payload: dict[str, Any], task_index: int) -> None:
    learning_rate, seed, condition = TASKS[task_index]
    identity = payload.get("identity", {})
    expected = {"task_index": task_index, "learning_rate": learning_rate, "seed": seed, "condition": condition, "train_per_stage": 10_000, "test_per_stage": 2_000, "class_exclusive": True, "classes_per_stage": 20}
    if payload.get("status") != "complete" or identity != expected:
        raise RuntimeError(f"identity/status mismatch for task {task_index}")
    if payload.get("condition") != condition or payload.get("seed") != seed or payload.get("training", {}).get("learning_rate") != learning_rate:
        raise RuntimeError(f"run payload mismatch for task {task_index}")
    if payload.get("classes_per_stage") != [20] * 5 or len(payload.get("class_groups", [])) != 100:
        raise RuntimeError(f"class grouping mismatch for task {task_index}")
    evaluation = payload.get("evaluation", {})
    records = evaluation.get("records", [])
    stream = payload.get("stream", {})
    if evaluation.get("expected_count") != 26 or len(records) != 26 or stream.get("length") != 50_000 or stream.get("stage_lengths") != [10_000] * 5:
        raise RuntimeError(f"stream/evaluation mismatch for task {task_index}")
    if [record["position"] for record in records] != list(range(0, 50_001, 2_000)):
        raise RuntimeError(f"evaluation positions mismatch for task {task_index}")
    test_counts = [2_000] * 5
    recomputed = continual_metrics(records, 50_000, stream["stage_order"], test_counts, 5)
    for key in METRICS:
        if not close(float(recomputed[key]), float(payload["summary"][key])):
            raise RuntimeError(f"summary mismatch task {task_index}: {key}")


def stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)), "values": array.tolist()}


def aggregate(root: Path) -> dict[str, Any]:
    runs = []
    for task_index in range(len(TASKS)):
        learning_rate, seed, condition = TASKS[task_index]
        path = output_path(root, learning_rate, seed, condition)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate(payload, task_index)
        runs.append(payload)
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(float(run["training"]["learning_rate"]), run["condition"])].append(run)
    results = {}
    for learning_rate in LEARNING_RATES:
        lr_result = {}
        for condition in CONDITIONS:
            condition_runs = sorted(grouped[(learning_rate, condition)], key=lambda item: item["seed"])
            entry = {metric: stats([float(run["summary"][metric]) for run in condition_runs]) for metric in METRICS}
            route_values = []
            stage_route_values = []
            if condition in ("random_routed_5", "online_routed_5", "oracle_routed_5"):
                for run in condition_runs:
                    route_values.append(float(run["evaluation"]["records"][-1]["routing"]["route_accuracy"]))
                    stage_route_values.append([float(run["evaluation"]["records"][(stage + 1) * 5]["routing"]["route_accuracy"]) for stage in range(5)])
                entry["final_routing_accuracy"] = stats(route_values)
                entry["stage_end_routing_accuracy"] = {"mean": np.asarray(stage_route_values).mean(0).tolist(), "std": np.asarray(stage_route_values).std(0, ddof=1).tolist(), "values": stage_route_values}
            entry["stage_seen_accuracy"] = {"mean": np.asarray([run["summary"]["stage_seen_accuracy"] for run in condition_runs]).mean(0).tolist(), "std": np.asarray([run["summary"]["stage_seen_accuracy"] for run in condition_runs]).std(0, ddof=1).tolist()}
            lr_result[condition] = entry
        results[f"{learning_rate:.8g}"] = lr_result
    comparisons = {}
    for learning_rate in LEARNING_RATES:
        key = f"{learning_rate:.8g}"
        comparisons[key] = {
            "online_minus_shared_auc": results[key]["online_routed_5"]["seen_anytime_auc"]["mean"] - results[key]["shared"]["seen_anytime_auc"]["mean"],
            "online_minus_random_auc": results[key]["online_routed_5"]["seen_anytime_auc"]["mean"] - results[key]["random_routed_5"]["seen_anytime_auc"]["mean"],
            "oracle_minus_shared_auc": results[key]["oracle_routed_5"]["seen_anytime_auc"]["mean"] - results[key]["shared"]["seen_anytime_auc"]["mean"],
            "online_beats_random_seeds": int(sum(a > b for a, b in zip(results[key]["online_routed_5"]["seen_anytime_auc"]["values"], results[key]["random_routed_5"]["seen_anytime_auc"]["values"], strict=True))),
        }
    auc_ratios = {}
    for learning_rate in LEARNING_RATES:
        key = f"{learning_rate:.8g}"
        auc_ratios[key] = {}
        for condition in CONDITIONS:
            best = max(results[f"{candidate:.8g}"][condition]["seen_anytime_auc"]["mean"] for candidate in LEARNING_RATES)
            auc_ratios[key][condition] = results[key][condition]["seen_anytime_auc"]["mean"] / best
    common_learning_rate = max(
        LEARNING_RATES,
        key=lambda candidate: (
            min(auc_ratios[f"{candidate:.8g}"].values()),
            float(np.mean(list(auc_ratios[f"{candidate:.8g}"].values()))),
            -candidate,
        ),
    )
    selection = {
        "criterion": "maximize the minimum across-condition ratio to each condition's best observed Seen-region Anytime AUC; ties use mean ratio then smaller learning rate",
        "selected_common_learning_rate": common_learning_rate,
        "auc_ratio_to_condition_best": auc_ratios,
        "minimum_ratio_by_learning_rate": {key: min(values.values()) for key, values in auc_ratios.items()},
        "best_learning_rate_by_condition": {condition: max(LEARNING_RATES, key=lambda candidate: results[f"{candidate:.8g}"][condition]["seen_anytime_auc"]["mean"]) for condition in CONDITIONS},
    }
    return {"schema_version": 1, "experiment": "olfactory_disjoint_baseline", "status": "complete", "audit": {"expected_runs": len(TASKS), "valid_runs": len(runs), "learning_rates": list(LEARNING_RATES), "seeds": list(SEEDS), "conditions": list(CONDITIONS), "class_exclusive": True, "classes_per_stage": 20, "train_per_stage": 10_000, "test_per_stage": 2_000}, "results": results, "comparisons": comparisons, "common_learning_rate_selection": selection}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=PROJECT_ROOT / "results" / "disjoint_baseline")
    args = parser.parse_args()
    payload = aggregate(args.result_root.resolve())
    atomic_json(args.result_root.resolve() / "analysis.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
