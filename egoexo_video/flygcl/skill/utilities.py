from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from flygcl.common.config import load_config, resolve_data_paths
from flygcl.common.data import ACTION_GROUPS, ACTION_TO_TASK, fit_bradley_terry_scores, read_pairs
from flygcl.common.metrics import summarize_accuracy_matrix


def load_prediction_rows(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        str(row["sample_id"]): row
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    }


def normalized(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.median(np.abs(vector))), 1e-8)


def build_graph_margins(config_path: Path, neighbors: int, temperature: float):
    config = load_config(config_path)
    paths = resolve_data_paths(config)
    train_pairs = read_pairs(paths["train_pairs"])
    val_pairs = read_pairs(paths["val_pairs"])
    scores = fit_bradley_terry_scores(
        train_pairs, iterations=500, learning_rate=0.08, l2=1e-3
    )
    cache: dict[str, np.ndarray] = {}

    def feature(video_id: str) -> np.ndarray:
        if video_id not in cache:
            with np.load(paths["ego_root"] / f"{video_id}.npz") as archive:
                tokens = archive["arr_0"].astype(np.float32, copy=False)
            if tokens.ndim == 3:
                tokens = tokens[0]
            mean = tokens.mean(0)
            trend = tokens[-1] - tokens[0]
            mean = mean / max(float(np.linalg.norm(mean)), 1e-8)
            trend = trend / max(float(np.linalg.norm(trend)), 1e-8)
            descriptor = np.concatenate((mean, 0.35 * trend))
            cache[video_id] = descriptor / max(float(np.linalg.norm(descriptor)), 1e-8)
        return cache[video_id]

    graph: dict[int, dict[str, float]] = {}
    for task_id in range(len(ACTION_GROUPS)):
        training_videos = sorted(
            {
                video
                for pair in train_pairs
                if ACTION_TO_TASK[pair.better.action] == task_id
                for video in (pair.better.raw, pair.worse.raw)
            }
        )
        pairs = [
            pair for pair in val_pairs if ACTION_TO_TASK[pair.better.action] == task_id
        ]
        validation_videos = sorted(
            {video for pair in pairs for video in (pair.better.raw, pair.worse.raw)}
        )
        train_features = np.stack([feature(video) for video in training_videos])
        train_scores = np.asarray([scores[video] for video in training_videos])
        query_features = np.stack([feature(video) for video in validation_videos])
        similarities = query_features @ train_features.T
        k = min(max(int(neighbors), 1), len(training_videos))
        indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
        selected = np.take_along_axis(similarities, indices, axis=1)
        weights = np.exp(
            (selected - selected.max(1, keepdims=True))
            / max(float(temperature), 1e-6)
        )
        predictions = (weights * train_scores[indices]).sum(1) / weights.sum(1)
        estimates = {
            video: scores[video] if video in scores else float(predictions[index])
            for index, video in enumerate(validation_videos)
        }
        graph[task_id] = {
            pair.sample_id: estimates[pair.better.raw] - estimates[pair.worse.raw]
            for pair in pairs
        }
    return graph


def prediction_path(root: Path, session: int, task: int) -> Path:
    return root / f"task_{session:02d}/predictions/eval_task_{task:02d}.jsonl"


def snapshot_records(run: Path, graph: dict, decay: float):
    records = []
    final_counts = []
    for session in range(1, 5):
        counts = []
        for task in range(1, session + 1):
            mappings = [
                load_prediction_rows(prediction_path(run, snapshot, task))
                for snapshot in range(task, session + 1)
            ]
            ids = sorted(mappings[0])
            if any(set(mapping) != set(ids) for mapping in mappings[1:]):
                raise RuntimeError(f"Snapshot sample mismatch: {run=} {session=} {task=}")
            weights = np.asarray(
                [math.exp(-float(decay) * (session - snapshot)) for snapshot in range(task, session + 1)]
            )
            margins = np.stack(
                [np.asarray([mapping[item]["margin"] for item in ids]) for mapping in mappings]
            )
            neural = np.average(margins, axis=0, weights=weights)
            graph_margin = np.asarray([graph[task - 1][item] for item in ids])
            current = mappings[-1]
            records.append(
                {
                    "session": session,
                    "task": task,
                    "ids": ids,
                    "base": 0.50 * normalized(neural) + 0.50 * normalized(graph_margin),
                    "routes": np.asarray(
                        [int(current[item]["routed_task"]) for item in ids], dtype=np.int64
                    ),
                }
            )
            counts.append(len(ids))
        if session == 4:
            final_counts = counts
    return records, final_counts


def record_metrics(records, counts):
    matrix = []
    for session in range(1, 5):
        matrix.append(
            [
                float((record["base"] > 0).mean())
                for record in records
                if record["session"] == session
            ]
        )
    return matrix, summarize_accuracy_matrix(matrix, counts)
