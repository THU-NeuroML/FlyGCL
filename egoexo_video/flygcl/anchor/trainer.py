from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common.data import (
    ACTION_GROUPS,
    ExoPools,
    SkillDataset,
    build_manifest,
    collate_records,
    read_pairs,
    write_json,
)
from ..common.metrics import summarize_accuracy_matrix
from .system import ContinualSystem


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class AnchorTrainer:
    """Train the fixed task-id-free TL anchor required by FlyGCL."""

    def __init__(self, config: Dict, paths: Dict[str, Path], output_dir: str | Path):
        self.config = config
        self.paths = paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(config.get("seed", 42))
        seed_everything(self.seed)
        requested = str(config.get("runtime", {}).get("device", "cuda"))
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.train_pairs = read_pairs(paths["train_pairs"])
        self.val_pairs = read_pairs(paths["val_pairs"])
        self.exo_pools = ExoPools(
            paths["exo_root"],
            seed=int(config.get("data", {}).get("exo_split_seed", 42)),
            val_ratio=float(config.get("data", {}).get("exo_val_ratio", 0.2)),
        )
        self.manifest = build_manifest(
            self.train_pairs, self.val_pairs, paths["ego_root"], self.exo_pools
        )
        write_json(self.output_dir / "task_manifest.json", self.manifest)
        if self.manifest["missing_ego_features"]:
            raise FileNotFoundError("Missing ego features; see task_manifest.json")
        self.system = ContinualSystem(config).to(self.device)
        training = config.get("training", {})
        self.optimizer = torch.optim.AdamW(
            self.system.parameters(),
            lr=float(training.get("lr", 2e-4)),
            weight_decay=float(training.get("weight_decay", 1e-6)),
        )
        self.start_task = 0
        self.matrix: List[List[float]] = []
        self.eval_counts: List[int] = []

    def dataset(self, split: str, task_id: int) -> SkillDataset:
        pairs = self.train_pairs if split == "train" else self.val_pairs
        return SkillDataset(
            pairs,
            self.paths["ego_root"],
            self.exo_pools,
            split,
            task_id,
            "ego_exo",
            int(self.config.get("model", {}).get("input_dim", 1024)),
        )

    def loader(self, split: str, task_id: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.dataset(split, task_id),
            batch_size=int(self.config.get("training", {}).get("batch_size", 32)),
            shuffle=shuffle,
            num_workers=int(self.config.get("runtime", {}).get("num_workers", 0)),
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_records,
            generator=torch.Generator().manual_seed(self.seed + task_id),
        )

    @staticmethod
    def _limited(loader: DataLoader, maximum: Optional[int]):
        for index, batch in enumerate(loader):
            if maximum is not None and index >= int(maximum):
                break
            yield batch

    def train_task(self, task_id: int) -> Dict[str, float]:
        self.system.begin_task(task_id)
        cfg = self.config.get("training", {})
        schedule = cfg.get("epochs_by_task")
        epochs = int(schedule[task_id]) if schedule else int(cfg.get("epochs_per_task", 1))
        totals: Dict[str, float] = {}
        steps = 0
        for _ in range(epochs):
            self.system.train()
            for raw in self._limited(
                self.loader("train", task_id, True),
                self.config.get("runtime", {}).get("max_train_batches"),
            ):
                batch = move_batch(raw, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss, details = self.system.loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at task {task_id + 1}")
                loss.backward()
                self.optimizer.step()
                self.system.flyprompt.enforce_frozen()
                for key, value in {"total": float(loss.detach()), **details}.items():
                    totals[key] = totals.get(key, 0.0) + value
                steps += 1
        return {key: value / max(steps, 1) for key, value in totals.items()}

    @torch.no_grad()
    def fit_router(self, task_id: int) -> None:
        self.system.eval()
        maximum = self.config.get("runtime", {}).get("max_router_batches")
        for raw in self._limited(self.loader("train", task_id), maximum):
            self.system.collect_router(move_batch(raw, self.device), task_id)
        self.system.update_router()

    @torch.no_grad()
    def evaluate_task(self, session_id: int, task_id: int) -> Tuple[float, int]:
        self.system.eval()
        rows = []
        correct = total = 0
        maximum = self.config.get("runtime", {}).get("max_eval_batches")
        for raw in self._limited(self.loader("val", task_id), maximum):
            sample_ids = list(raw["sample_id"])
            batch = move_batch(raw, self.device)
            result = self.system.forward_skill(batch, training=False)
            margins = result["score_better"] - result["score_worse"]
            routes = result["routed_task"]
            correct += int(margins.gt(0).sum())
            total += int(margins.numel())
            for sample_id, margin, route in zip(sample_ids, margins.cpu(), routes.cpu()):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "y_true": 1,
                        "y_pred": int(float(margin) > 0),
                        "margin": float(margin),
                        "routed_task": int(route),
                    }
                )
        path = self.output_dir / f"task_{session_id + 1:02d}/predictions/eval_task_{task_id + 1:02d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return correct / max(total, 1), total

    def save_checkpoint(self, task_id: int) -> None:
        path = self.output_dir / f"task_{task_id + 1:02d}/checkpoint.pt"
        state = self.system.state_dict()
        state.pop("flyprompt.router.gram", None)
        state.pop("flyprompt.router.targets", None)
        torch.save(
            {
                "completed_task": task_id,
                "model": state,
                "accuracy_matrix": self.matrix,
                "eval_counts": self.eval_counts,
            },
            path,
        )

    def rebuild_router(self, completed_tasks: int) -> None:
        router = self.system.flyprompt.router
        router.gram.zero_()
        router.targets.zero_()
        router.seen_tasks.zero_()
        self.system.flyprompt.prototype_sums.zero_()
        self.system.flyprompt.prototype_counts.zero_()
        for task_id in range(completed_tasks):
            for raw in self.loader("train", task_id):
                self.system.collect_router(move_batch(raw, self.device), task_id)
        self.system.update_router()

    def resume(self, checkpoint: str | Path) -> None:
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        loaded = self.system.load_state_dict(state["model"], strict=False)
        allowed = {"flyprompt.router.gram", "flyprompt.router.targets"}
        if set(loaded.missing_keys) - allowed or loaded.unexpected_keys:
            raise RuntimeError(f"Checkpoint mismatch: {loaded}")
        self.matrix = [list(map(float, row)) for row in state.get("accuracy_matrix", [])]
        self.eval_counts = list(map(int, state.get("eval_counts", [])))
        self.start_task = int(state["completed_task"]) + 1
        self.rebuild_router(self.start_task)

    def run(self, resume: Optional[str | Path] = None) -> Dict[str, object]:
        if resume:
            self.resume(resume)
        for task_id in range(self.start_task, len(ACTION_GROUPS)):
            losses = self.train_task(task_id)
            self.fit_router(task_id)
            row, counts = [], []
            for eval_task in range(task_id + 1):
                accuracy, count = self.evaluate_task(task_id, eval_task)
                row.append(accuracy)
                counts.append(count)
            self.matrix.append(row)
            if len(counts) > len(self.eval_counts):
                self.eval_counts = counts
            write_json(
                self.output_dir / f"task_{task_id + 1:02d}/metrics_task_end.json",
                {"task": task_id + 1, "train_loss": losses, "seen_task_accuracy": row},
            )
            self.save_checkpoint(task_id)
        result = {
            "status": "complete",
            "benchmark": "skill",
            "view": "ego_exo",
            "skill_head": "tl",
            "method": "flygcl_tl_anchor",
            "task_order": list(ACTION_GROUPS),
            "accuracy_matrix": self.matrix,
            "metrics": summarize_accuracy_matrix(self.matrix, self.eval_counts),
        }
        write_json(self.output_dir / "final_results.json", result)
        return result


def copy_resolved_config(config: Dict, output_dir: str | Path) -> None:
    write_json(
        Path(output_dir) / "resolved_config.json",
        {key: value for key, value in config.items() if key != "config_path"},
    )
