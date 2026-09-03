#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flygcl.common.config import load_config, resolve_data_paths
from flygcl.common.data import (
    ExoPools,
    SkillDataset,
    collate_records,
    fit_bradley_terry_scores,
    read_pairs,
)
from flygcl.anchor.trainer import move_batch, seed_everything
from flygcl.skill.tl_model import TemporalCrossViewExpert, ranking_objective
from flygcl.skill.utilities import load_prediction_rows


def prediction_path(root: Path, session: int, task: int) -> Path:
    return root / f"task_{session:02d}/predictions/eval_task_{task:02d}.jsonl"


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_records,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def augment(value: torch.Tensor, token_dropout: float, noise_std: float) -> torch.Tensor:
    keep = (torch.rand(value.shape[:-1] + (1,), device=value.device) >= token_dropout).to(value.dtype)
    return value * keep + torch.randn_like(value) * noise_std


@torch.no_grad()
def evaluate_session(
    experts: list[TemporalCrossViewExpert],
    val_pairs,
    paths: Dict[str, Path],
    exo_pools: ExoPools,
    session: int,
    base_run: Path,
    output: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> None:
    for expert in experts:
        expert.eval()
    for task in range(session):
        dataset = SkillDataset(
            val_pairs,
            paths["ego_root"],
            exo_pools,
            "val",
            task,
            "ego_exo",
            exo_references=4,
        )
        route_rows = load_prediction_rows(prediction_path(base_run, session, task + 1))
        destination = prediction_path(output, session, task + 1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for batch in loader(dataset, batch_size, workers, False, 10000 + session * 10 + task):
            sample_ids = list(batch["sample_id"])
            routes = torch.tensor(
                [int(route_rows[item]["routed_task"]) for item in sample_ids],
                device=device,
                dtype=torch.long,
            )
            if int(routes.max()) >= session:
                raise RuntimeError(f"Future route at session {session}: {int(routes.max())}")
            batch = move_batch(batch, device)
            margins = torch.zeros(len(sample_ids), device=device)
            for route in routes.unique(sorted=True):
                indices = torch.nonzero(routes == route, as_tuple=False).flatten()
                result = experts[int(route)].forward_pair(
                    batch["better"].index_select(0, indices),
                    batch["worse"].index_select(0, indices),
                    batch["exo"].index_select(0, indices),
                )
                margins.index_copy_(0, indices, result["margin"])
            for sample_id, route, value in zip(sample_ids, routes.cpu(), margins.cpu()):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "y_true": 1,
                        "y_pred": int(float(value) > 0),
                        "margin": float(value),
                        "routed_task": int(route),
                    }
                )
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="FlyGCL detached cross-view TL expert")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", nargs=4, type=int, default=(12, 16, 10, 8))
    parser.add_argument("--auto-resume", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError(f"Output must remain inside {ROOT}: {output}")
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "expert_results.json"
    if args.auto_resume and final_path.is_file():
        prior = json.loads(final_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            print(f"[skip] completed: {output}")
            return 0
    base_run = Path(args.base_run).resolve()
    config = load_config(args.config)
    paths = resolve_data_paths(config)
    device_name = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    train_pairs = read_pairs(paths["train_pairs"])
    val_pairs = read_pairs(paths["val_pairs"])
    exo_pools = ExoPools(
        paths["exo_root"],
        seed=int(config.get("data", {}).get("exo_split_seed", 42)),
        val_ratio=float(config.get("data", {}).get("exo_val_ratio", 0.2)),
    )
    skill_scores = fit_bradley_terry_scores(train_pairs, iterations=500)
    experts: list[TemporalCrossViewExpert] = []
    training_log = []
    checkpoint = output / "checkpoint.pt"
    if args.auto_resume and checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("method") == "flygcl_cross_view_expert":
            for expert_state in state.get("experts", []):
                restored = TemporalCrossViewExpert().to(device)
                restored.load_state_dict(expert_state)
                restored.eval()
                restored.requires_grad_(False)
                experts.append(restored)
            training_log = list(state.get("training_log", []))
            print(f"[resume] restored {len(experts)} completed task experts", flush=True)
    for task in range(len(experts), 4):
        expert = TemporalCrossViewExpert().to(device)
        if experts:
            expert.load_state_dict(experts[-1].state_dict())
        dataset = SkillDataset(
            train_pairs,
            paths["ego_root"],
            exo_pools,
            "train",
            task,
            "ego_exo",
            exo_references=4,
            skill_scores=skill_scores,
        )
        optimizer = torch.optim.AdamW(expert.parameters(), lr=2e-4, weight_decay=5e-4)
        final_details = {}
        for epoch in range(int(args.epochs[task])):
            expert.train()
            totals: Dict[str, float] = {}
            count = 0
            for batch in loader(
                dataset,
                args.batch_size,
                args.num_workers,
                True,
                args.seed + task * 1009 + epoch,
            ):
                batch = move_batch(batch, device)
                better = augment(batch["better"], 0.10, 0.01)
                worse = augment(batch["worse"], 0.10, 0.01)
                exo = augment(batch["exo"], 0.08, 0.008)
                optimizer.zero_grad(set_to_none=True)
                result = expert.forward_pair(better, worse, exo)
                loss, details = ranking_objective(
                    result,
                    batch["better_skill_target"],
                    batch["worse_skill_target"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite FlyGCL loss at task {task + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(expert.parameters(), 5.0)
                optimizer.step()
                for key, value in details.items():
                    totals[key] = totals.get(key, 0.0) + value
                count += 1
            final_details = {key: value / max(count, 1) for key, value in totals.items()}
            print(
                f"[train] task={task + 1} epoch={epoch + 1}/{args.epochs[task]} "
                f"loss={final_details.get('loss', float('nan')):.5f}",
                flush=True,
            )
        expert.eval()
        expert.requires_grad_(False)
        experts.append(expert)
        training_log.append({"task": task + 1, "epochs": args.epochs[task], **final_details})
        evaluate_session(
            experts,
            val_pairs,
            paths,
            exo_pools,
            task + 1,
            base_run,
            output,
            device,
            args.batch_size,
            args.num_workers,
        )
        atomic_torch_save(
            {
                "method": "flygcl_cross_view_expert",
                "seed": args.seed,
                "completed_task": task,
                "experts": [item.state_dict() for item in experts],
                "training_log": training_log,
            },
            checkpoint,
        )
    payload = {
        "status": "complete",
        "method": "FlyGCL detached multi-scale exo-conditioned TL expert",
        "seed": args.seed,
        "epochs_by_task": list(args.epochs),
        "training_log": training_log,
        "routing": "FlyGCL task-id-free routed_task; no oracle task label used at evaluation",
    }
    final_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
