"""Conserved Si-Blurry streams over the immutable formal subset."""
from __future__ import annotations
import hashlib
from typing import Any
import numpy as np
from .config import Config
from .data import formal_subset

CELLS = ((50, 10), (50, 50), (0, 10), (0, 50))


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def build_blurry_subset(seed: int, n_disjoint: int, m_blurry: int, cfg: Config) -> dict[str, Any]:
    if (n_disjoint, m_blurry) not in CELLS:
        raise ValueError("unsupported boundary-blur cell")
    dataset = formal_subset(seed, cfg)
    labels = np.asarray(dataset["train_labels"], dtype=np.int64)
    home = np.asarray(dataset["train_regions"], dtype=np.int64)
    class_groups = np.asarray(dataset["class_groups"], dtype=np.int64)
    n_disjoint_per_stage = 20 * n_disjoint // 100

    selection_rng = np.random.default_rng(np.random.SeedSequence([2026, 8, 24, 510_000, seed, n_disjoint]))
    disjoint_classes: list[np.ndarray] = []
    blurry_classes: list[np.ndarray] = []
    for stage in range(cfg.n_regions):
        classes = np.flatnonzero(class_groups == stage)
        if classes.size != 20:
            raise RuntimeError("home stage must contain exactly 20 classes")
        classes = selection_rng.permutation(classes)
        disjoint_classes.append(np.sort(classes[:n_disjoint_per_stage]))
        blurry_classes.append(np.sort(classes[n_disjoint_per_stage:]))

    pool_parts: list[np.ndarray] = []
    kept_parts: list[np.ndarray] = []
    quotas: list[int] = []
    for stage in range(cfg.n_regions):
        stage_indices = np.flatnonzero(home == stage)
        is_blurry = np.isin(labels[stage_indices], blurry_classes[stage])
        blurry_indices = stage_indices[is_blurry]
        fixed_indices = stage_indices[~is_blurry]
        extract_rng = np.random.default_rng(np.random.SeedSequence([2026, 8, 24, 520_000, seed, n_disjoint, stage]))
        blurry_indices = extract_rng.permutation(blurry_indices)
        n_out = blurry_indices.size * m_blurry // 100
        pool_parts.append(blurry_indices[:n_out])
        kept_parts.append(np.concatenate((fixed_indices, blurry_indices[n_out:])))
        quotas.append(n_out)

    pool = np.concatenate(pool_parts)
    pool_rng = np.random.default_rng(np.random.SeedSequence([2026, 8, 24, 530_000, seed, n_disjoint, m_blurry]))
    pool = pool_rng.permutation(pool)
    stage_indices: list[np.ndarray] = []
    cursor = 0
    for stage, quota in enumerate(quotas):
        received = pool[cursor:cursor + quota]
        cursor += quota
        merged = np.concatenate((kept_parts[stage], received))
        shuffle_rng = np.random.default_rng(np.random.SeedSequence([2026, 8, 24, 540_000, seed, n_disjoint, m_blurry, stage]))
        stage_indices.append(shuffle_rng.permutation(merged))
    if cursor != pool.size:
        raise RuntimeError("pooled sample allocation mismatch")

    order = np.concatenate(stage_indices).astype(np.int64, copy=False)
    stage_lengths = [int(indices.size) for indices in stage_indices]
    expected = np.arange(labels.size, dtype=np.int64)
    if labels.size != 50_000 or stage_lengths != [cfg.train_per_stage] * cfg.n_regions:
        raise RuntimeError("boundary-blur stage length mismatch")
    if not np.array_equal(np.sort(order), expected):
        raise RuntimeError("boundary-blur stream must be a complete unique permutation")
    arrival = np.repeat(np.arange(cfg.n_regions, dtype=np.int64), stage_lengths)
    stream_labels = labels[order]
    exposure = np.zeros(cfg.n_classes, dtype=np.int64)
    first_exposure = np.full(cfg.n_classes, -1, dtype=np.int64)
    for position, label in enumerate(stream_labels, start=1):
        exposure[label] += 1
        if first_exposure[label] < 0:
            first_exposure[label] = position
    if np.any(exposure == 0):
        raise RuntimeError("every class must occur in the formal stream")

    metadata = {
        "cell": [n_disjoint, m_blurry],
        "stage_lengths": stage_lengths,
        "stage_boundaries": np.cumsum(stage_lengths).astype(int).tolist(),
        "disjoint_classes": [value.tolist() for value in disjoint_classes],
        "blurry_classes": [value.tolist() for value in blurry_classes],
        "first_exposure_position": first_exposure.tolist(),
        "stream_order_sha256": array_sha256(order),
        "stream_labels_sha256": array_sha256(stream_labels),
        "arrival_stages_sha256": array_sha256(arrival),
        "sample_conservation": True,
        "unique_samples": True,
    }
    return {**dataset, "stream_indices": order, "arrival_stages": arrival, "first_exposure_position": first_exposure, "blur_metadata": metadata}
