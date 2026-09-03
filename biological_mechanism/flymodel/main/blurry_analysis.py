"""Deep audit and aggregate boundary-blur results."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from .audit import atomic_json, canonical_sha256
from .blurry_experiment import CONDITIONS
from .blurry_metrics import exposure_metrics
from .config import READOUTS, SEEDS
from .blurry_run import DEFAULT_ROOT, TASKS, expected_identity, output_path, source_identity


def close(a: float, b: float) -> bool: return bool(np.isclose(a, b, rtol=1e-8, atol=1e-9))


def stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float); return {"mean": float(array.mean()), "std": float(array.std(ddof=1)), "values": array.tolist()}


def validate(payload: dict[str, Any], index: int) -> None:
    task = TASKS[index]; identity = expected_identity(index, task)
    if payload.get("status") != "complete" or payload.get("experiment") != "olfactory_boundary_blur" or payload.get("identity") != identity: raise RuntimeError(f"identity mismatch task {index}")
    training = payload.get("training", {})
    if training.get("eta") != 1e-3 or training.get("gamma") != 10 or training.get("rates") != identity["rates"] or training.get("optimizer") != "Adam" or training.get("recorded_integrations") != list(READOUTS): raise RuntimeError(f"training mismatch task {index}")
    stream = payload.get("stream", {}); records = payload.get("evaluation", {}).get("records", [])
    if stream.get("length") != 50_000 or stream.get("stage_lengths") != [10_000] * 5 or not stream.get("sample_conservation") or not stream.get("unique_samples") or len(records) != 26 or [record["position"] for record in records] != list(range(0, 50_001, 2_000)): raise RuntimeError(f"stream mismatch task {index}")
    fingerprint = payload.get("fingerprint", {}); configuration = fingerprint.get("configuration", {})
    if fingerprint.get("sha256") != canonical_sha256(configuration) or configuration.get("source", {}).get("sha256") != source_identity()["sha256"]: raise RuntimeError(f"fingerprint mismatch task {index}")
    for record in records:
        if set(record.get("readouts", {})) != set(READOUTS) or not record.get("audit", {}).get("read_only"): raise RuntimeError(f"record mismatch task {index}")
    for readout in READOUTS:
        selected = [{**record, "class_correct": record["readouts"][readout]["class_correct"], "class_test_counts": record["readouts"][readout]["class_test_counts"]} for record in records]
        recomputed = exposure_metrics(selected, 50_000, 100); summary = payload["readout_summaries"][readout]
        for key in ("exposed_anytime_auc", "final_accuracy", "average_class_forgetting", "worst_class_accuracy"):
            if not close(recomputed[key], summary[key]): raise RuntimeError(f"summary mismatch task {index} {readout} {key}")
    routed = task.condition.startswith("inherited_"); inheritance = payload.get("inheritance", {}); events = inheritance.get("events", [])
    if routed:
        if not inheritance.get("enabled") or len(events) != 4 or not payload.get("audits", {}).get("old_experts_immutable"): raise RuntimeError(f"inheritance mismatch task {index}")
        snapshots = payload.get("stage_parameter_snapshots", [])
        if len(snapshots) != 5: raise RuntimeError(f"snapshot mismatch task {index}")
        for stage in range(1, 5):
            for old in range(stage):
                if snapshots[stage]["hashes"][old] != snapshots[stage - 1]["hashes"][old]: raise RuntimeError(f"old expert mutation task {index}")
    elif inheritance.get("enabled") or events: raise RuntimeError(f"unexpected inheritance task {index}")


def aggregate(root: Path) -> dict[str, Any]:
    runs = []
    for index, task in enumerate(TASKS):
        path = output_path(root, task)
        if not path.is_file(): raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8")); validate(payload, index); runs.append(payload)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        cell = f"n{run['identity']['cell'][0]}m{run['identity']['cell'][1]}"; grouped[(cell, run["condition"])].append(run)
    results = {}
    for cell in ("n50m10", "n50m50", "n0m10", "n0m50"):
        cell_results = {}
        for condition in CONDITIONS:
            ordered = sorted(grouped[(cell, condition)], key=lambda item: item["seed"]); entry = {}
            for readout in READOUTS:
                entry[readout] = {metric: stats([float(run["readout_summaries"][readout][metric]) for run in ordered]) for metric in ("exposed_anytime_auc", "final_accuracy", "average_class_forgetting", "worst_class_accuracy")}
            if condition.startswith("inherited_"):
                final_records = [run["evaluation"]["records"][-1] for run in ordered]
                entry["routing"] = {
                    "home_region_accuracy": stats([float(record["routing"]["home_region_accuracy"]) for record in final_records]),
                    "wrong_route_classification_accuracy": stats([float(record["readouts"]["softmax_mean"]["wrong_route_classification_accuracy"]) for record in final_records]),
                    "outside_selected_expert_prediction_rate": stats([float(record["readouts"]["softmax_mean"]["outside_selected_expert_prediction_rate"]) for record in final_records]),
                }
            cell_results[condition] = entry
        primary = {name: cell_results[name]["softmax_mean"]["final_accuracy"]["values"] for name in CONDITIONS}
        comparisons = {}
        for name, left, right in (("el_minus_single", "shared_el", "single_head"), ("moe_minus_single", "inherited_moe_mid", "single_head"), ("moe_el_minus_moe", "inherited_moe_el", "inherited_moe_mid"), ("moe_el_minus_el", "inherited_moe_el", "shared_el"), ("full_minus_single", "inherited_moe_el", "single_head")):
            delta = np.asarray(primary[left]) - np.asarray(primary[right]); comparisons[name] = {**stats(delta.tolist()), "positive_seeds": int((delta > 0).sum())}
        results[cell] = {"conditions": cell_results, "comparisons": comparisons}
    return {"schema_version": 1, "experiment": "olfactory_boundary_blur", "status": "complete", "audit": {"expected_runs": 80, "valid_runs": len(runs), "cells": [[50, 10], [50, 50], [0, 10], [0, 50]], "conditions": list(CONDITIONS), "seeds": list(SEEDS), "eta": 1e-3, "gamma": 10, "readouts": list(READOUTS)}, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT); args = parser.parse_args(); root = args.result_root.resolve(); payload = aggregate(root); atomic_json(root / "analysis.json", payload); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
