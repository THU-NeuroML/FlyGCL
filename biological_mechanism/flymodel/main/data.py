"""Loader for immutable olfactory assets and class-exclusive subsets."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment
from .audit import canonical_sha256, file_sha256
from .config import Config


def balanced_prototype_groups(prototypes: np.ndarray, n_groups: int, seed: int) -> np.ndarray:
    if len(prototypes) % n_groups:
        raise ValueError("prototype count must be divisible by group count")
    capacity = len(prototypes) // n_groups
    first = int(np.argmax(np.sum((prototypes - prototypes.mean(0)) ** 2, axis=1)))
    chosen = [first]; nearest = np.full(len(prototypes), np.inf)
    while len(chosen) < n_groups:
        nearest = np.minimum(nearest, np.sum((prototypes - prototypes[chosen[-1]]) ** 2, axis=1)); nearest[chosen] = -1; chosen.append(int(np.argmax(nearest)))
    centers = prototypes[chosen].copy(); assignment = np.zeros(len(prototypes), dtype=np.uint8)
    for _ in range(30):
        costs = np.sum((prototypes[:, None, :] - centers[None, :, :]) ** 2, axis=2); slots = np.repeat(np.arange(n_groups), capacity); _, columns = linear_sum_assignment(costs[:, slots]); updated = slots[columns].astype(np.uint8); new_centers = np.stack([prototypes[updated == group].mean(0) for group in range(n_groups)]); assignment = updated
        if np.allclose(new_centers, centers): break
        centers = new_centers
    if not np.array_equal(np.bincount(assignment, minlength=n_groups), np.full(n_groups, capacity)):
        raise RuntimeError("balanced grouping failed")
    return assignment


def load_v4_seed(seed: int, cfg: Config) -> dict[str, Any]:
    root = cfg.data_root / f"seed_{seed}"; metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")); fingerprint = metadata.pop("fingerprint", None)
    if metadata.get("schema_version") != 2 or metadata.get("experiment") != "olfactory_fixed_encoder_assets" or metadata.get("status") != "complete" or metadata.get("seed") != seed or fingerprint != canonical_sha256(metadata):
        raise RuntimeError("olfactory data identity/fingerprint mismatch")
    metadata["fingerprint"] = fingerprint
    expected = {"prototypes":((cfg.n_classes,cfg.odor_dim),"float32"),"train_samples":((1_000_000,cfg.odor_dim),"float32"),"test_samples":((200_000,cfg.odor_dim),"float32"),"train_labels":((1_000_000,),"uint8"),"test_labels":((200_000,),"uint8"),"orn_pn":((cfg.n_pn,cfg.orn_per_channel),"float32"),"pn_kc":((cfg.n_pn,cfg.n_kc),"float32")}
    arrays = {}
    for key,(shape,dtype) in expected.items():
        name=f"{key}.npy"; path=root/name; record=metadata["files"][name]
        if file_sha256(path) != record["sha256"]: raise RuntimeError(f"data artifact hash mismatch: {name}")
        array=np.load(path,mmap_mode="r",allow_pickle=False)
        if array.shape != shape or str(array.dtype) != dtype: raise RuntimeError(f"data shape/dtype mismatch: {name}")
        arrays[key]=array
    return {"root":root,"metadata":metadata,**arrays}


def formal_subset(seed: int, cfg: Config) -> dict[str, Any]:
    dataset=load_v4_seed(seed,cfg); groups=balanced_prototype_groups(np.asarray(dataset["prototypes"]),cfg.n_regions,seed); train_regions=groups[np.asarray(dataset["train_labels"],dtype=np.int64)]; test_regions=groups[np.asarray(dataset["test_labels"],dtype=np.int64)]; rng=np.random.default_rng(np.random.SeedSequence([2026,8,23,430_000,seed]))
    train_idx=np.concatenate([rng.choice(np.flatnonzero(train_regions==group),cfg.train_per_stage,replace=False) for group in range(cfg.n_regions)]); test_idx=np.concatenate([rng.choice(np.flatnonzero(test_regions==group),cfg.test_per_stage,replace=False) for group in range(cfg.n_regions)])
    return {"train_samples":np.asarray(dataset["train_samples"][train_idx]),"train_labels":np.asarray(dataset["train_labels"][train_idx]),"train_regions":train_regions[train_idx],"test_samples":np.asarray(dataset["test_samples"][test_idx]),"test_labels":np.asarray(dataset["test_labels"][test_idx]),"test_regions":test_regions[test_idx],"stage_order":np.arange(cfg.n_regions,dtype=np.uint8),"orn_pn":np.asarray(dataset["orn_pn"]),"pn_kc":np.asarray(dataset["pn_kc"]),"class_groups":groups,"metadata":{"source_fingerprint":dataset["metadata"]["fingerprint"],"source_metadata_sha256":file_sha256(dataset["root"] / "metadata.json"),"counts":{"train":[cfg.train_per_stage]*cfg.n_regions,"test":[cfg.test_per_stage]*cfg.n_regions}}}
