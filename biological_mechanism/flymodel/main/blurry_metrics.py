"""Exposure-aware continual metrics for boundary-blurred streams."""
from __future__ import annotations
import numpy as np


def exposure_metrics(records: list[dict], stream_length: int, n_classes: int) -> dict:
    positions = np.asarray([record["position"] for record in records], dtype=float)
    exposure = np.asarray([record["exposure_counts"] for record in records], dtype=np.int64)
    correct = np.asarray([record["class_correct"] for record in records], dtype=float)
    counts = np.asarray(records[0]["class_test_counts"], dtype=float)
    if exposure.shape != (len(records), n_classes) or correct.shape != (len(records), n_classes):
        raise RuntimeError("invalid exposure metric shape")
    if np.any(counts <= 0) or not np.all(np.diff(exposure, axis=0) >= 0) or np.any(exposure[-1] == 0):
        raise RuntimeError("invalid class exposure or test counts")
    class_accuracy = correct / counts[None, :]
    exposed_accuracy: list[float | None] = []
    for row, seen in zip(class_accuracy, exposure > 0, strict=True):
        exposed_accuracy.append(float(np.average(row[seen], weights=counts[seen])) if np.any(seen) else None)
    valid = np.flatnonzero([value is not None for value in exposed_accuracy])
    if valid.size < 2:
        raise RuntimeError("insufficient exposed checkpoints")
    start = int(valid[0]); values = np.asarray([exposed_accuracy[index] for index in valid], dtype=float)
    auc = float(np.trapezoid(values, positions[valid]) / (positions[valid[-1]] - positions[valid[0]]))
    forgetting = []
    for label in range(n_classes):
        eligible = np.flatnonzero(exposure[:, label] > 0)
        forgetting.append(float(class_accuracy[eligible, label].max() - class_accuracy[-1, label]))
    final_accuracy = float(np.average(class_accuracy[-1], weights=counts))
    return {
        "exposed_anytime_auc": auc,
        "auc_start_position": int(positions[start]),
        "auc_end_position": int(positions[valid[-1]]),
        "exposed_accuracy": exposed_accuracy,
        "exposed_class_count": (exposure > 0).sum(1).astype(int).tolist(),
        "final_accuracy": final_accuracy,
        "average_class_forgetting": float(np.average(forgetting, weights=counts)),
        "class_forgetting": forgetting,
        "worst_class_accuracy": float(class_accuracy[-1].min()),
    }
