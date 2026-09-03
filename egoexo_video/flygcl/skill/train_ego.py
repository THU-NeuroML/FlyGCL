#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flygcl.common.config import load_config, resolve_data_paths
from flygcl.common.data import (
    ACTION_GROUPS,
    ExoPools,
    SkillDataset,
    collate_records,
    fit_bradley_terry_scores,
    read_pairs,
)
from flygcl.anchor.trainer import move_batch, seed_everything
from flygcl.skill.ego_model import (
    EgoTemporalMoE,
    clone_frozen,
    ranking_objective,
    update_ema,
)


def atomic_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_records,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def augment(features: torch.Tensor, dropout: float, noise: float) -> torch.Tensor:
    keep = (
        torch.rand(features.shape[:-1] + (1,), device=features.device) >= float(dropout)
    ).to(features.dtype)
    return features * keep + float(noise) * torch.randn_like(features)


def pair_query(batch: Dict[str, object]) -> torch.Tensor:
    query = 0.5 * (
        batch["better"].mean(1) + batch["worse"].mean(1)  # type: ignore[union-attr]
    )
    return F.normalize(query, dim=-1)


def dataset_for(
    pairs,
    paths,
    exo_pools,
    task: int,
    split: str,
    skill_scores,
    synthetic_ratio: float,
    seed: int,
):
    return SkillDataset(
        pairs,
        paths["ego_root"],
        exo_pools,
        split,
        task,
        "ego",
        skill_scores=skill_scores if split == "train" else None,
        synthetic_pair_ratio=float(synthetic_ratio) if split == "train" else 0.0,
        synthetic_pair_seed=int(seed),
        synthetic_minimum_gap=0.70,
    )


@torch.no_grad()
def task_prototype(loader: DataLoader, device: torch.device) -> torch.Tensor:
    total = torch.zeros(1024, device=device)
    count = 0
    for raw in loader:
        batch = move_batch(raw, device)
        query = pair_query(batch)
        total.add_(query.sum(0))
        count += query.shape[0]
    return F.normalize(total / max(count, 1), dim=0).cpu()


def train_loss(
    model: EgoTemporalMoE,
    batch: Dict[str, object],
    token_dropout: float,
    noise_std: float,
    consistency_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    better = batch["better"]  # type: ignore[assignment]
    worse = batch["worse"]  # type: ignore[assignment]
    augmented = model.forward_pair(
        augment(better, token_dropout, noise_std),
        augment(worse, token_dropout, noise_std),
    )
    loss, details = ranking_objective(augmented, batch["skill_gap_target"])
    if float(consistency_weight) > 0:
        with torch.no_grad():
            clean_margin = model.forward_pair(better, worse)["margin"]
        consistency = F.smooth_l1_loss(augmented["margin"], clean_margin)
        loss = loss + float(consistency_weight) * consistency
        details["consistency"] = float(consistency.detach())
    return loss, details


@torch.no_grad()
def evaluate_session(
    experts: list[EgoTemporalMoE],
    global_snapshot: EgoTemporalMoE,
    prototypes: torch.Tensor,
    val_pairs,
    paths,
    exo_pools,
    output: Path,
    session: int,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> None:
    for model in [*experts, global_snapshot]:
        model.eval()
    seen_prototypes = F.normalize(prototypes[:session].to(device), dim=-1)
    for task in range(session):
        dataset = dataset_for(
            val_pairs, paths, exo_pools, task, "val", {}, 0.0, 10000 + session + task
        )
        destination = (
            output
            / f"task_{session:02d}"
            / "predictions"
            / f"eval_task_{task + 1:02d}.jsonl"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        loader = make_loader(dataset, batch_size, workers, False, 10000 + session * 10 + task)
        for raw in loader:
            sample_ids = list(raw["sample_id"])
            batch = move_batch(raw, device)
            similarities = pair_query(batch) @ seen_prototypes.T
            probabilities = torch.softmax(similarities / 0.05, dim=-1)
            confidence, routes = probabilities.max(dim=-1)
            specialist = torch.zeros(len(sample_ids), device=device)
            consensus = torch.zeros(len(sample_ids), device=device)
            for route in routes.unique(sorted=True):
                indices = torch.nonzero(routes == route, as_tuple=False).flatten()
                result = experts[int(route)].forward_pair(
                    batch["better"].index_select(0, indices),
                    batch["worse"].index_select(0, indices),
                )
                specialist.index_copy_(0, indices, result["margin"])
                consensus.index_copy_(0, indices, result["expert_margins"].mean(-1))
            global_margin = global_snapshot.forward_pair(
                batch["better"], batch["worse"]
            )["margin"]
            for sample_id, route, conf, spec, cons, glob in zip(
                sample_ids,
                routes.cpu(),
                confidence.cpu(),
                specialist.cpu(),
                consensus.cpu(),
                global_margin.cpu(),
            ):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "y_true": 1,
                        "routed_task": int(route),
                        "route_confidence": float(conf),
                        "specialist_margin": float(spec),
                        "expert_consensus_margin": float(cons),
                        "global_margin": float(glob),
                    }
                )
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="FlyGCL enhanced Ego-only temporal MoE")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--auto-resume", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError(f"Output must remain inside {ROOT}: {output}")
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "training_complete.json"
    if args.auto_resume and final_path.is_file():
        payload = json.loads(final_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(f"[skip] completed: {output}")
            return 0

    config = load_config(args.config)
    paths = resolve_data_paths(config)
    cfg = config["skill_ego"]
    ema_decay = float(cfg.get("ema_decay", 0.985))
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("EMA decay must be in [0,1)")
    specialist_lr = float(cfg.get("learning_rate", 2e-4))
    global_lr = float(cfg.get("global_learning_rate", 1.2e-4))
    effective_weight_decay = float(cfg.get("weight_decay", 5e-4))
    training_overrides = {
        "specialist_learning_rate": specialist_lr,
        "global_learning_rate": global_lr,
        "weight_decay": effective_weight_decay,
        "ema_decay": ema_decay,
        "validation_used_for_training_or_checkpoint_selection": False,
    }
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_pairs = read_pairs(paths["train_pairs"])
    val_pairs = read_pairs(paths["val_pairs"])
    exo_pools = ExoPools(
        paths["exo_root"],
        seed=int(config.get("data", {}).get("exo_split_seed", 42)),
        val_ratio=float(config.get("data", {}).get("exo_val_ratio", 0.2)),
    )
    scores = fit_bradley_terry_scores(
        train_pairs,
        iterations=int(cfg.get("bt_iterations", 500)),
        learning_rate=0.08,
        l2=1e-3,
    )
    model_args = {
        "input_dim": int(config.get("model", {}).get("input_dim", 1024)),
        "hidden_dim": int(cfg.get("hidden_dim", 256)),
        "tokens": int(config.get("model", {}).get("num_tokens", 10)),
        "dropout": float(cfg.get("dropout", 0.12)),
    }
    global_model = EgoTemporalMoE(**model_args).to(device)
    global_ema = clone_frozen(global_model).to(device)
    experts: list[EgoTemporalMoE] = []
    prototypes = torch.zeros(len(ACTION_GROUPS), model_args["input_dim"])
    training_log = []
    start_task = 0
    checkpoint = output / "checkpoint.pt"
    if args.auto_resume and checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("method") == "flygcl_ego_temporal_moe_v1":
            global_model.load_state_dict(state["global_model"])
            global_ema.load_state_dict(state["global_ema"])
            for expert_state in state["experts"]:
                expert = EgoTemporalMoE(**model_args).to(device)
                expert.load_state_dict(expert_state)
                expert.eval().requires_grad_(False)
                experts.append(expert)
            prototypes.copy_(state["prototypes"])
            training_log = list(state.get("training_log", []))
            start_task = len(experts)
            print(f"[resume] restored {start_task} completed experts", flush=True)

    epochs = [max(1, int(value)) for value in cfg.get("epochs_by_task", [8, 10, 6, 4])]
    for task in range(start_task, len(ACTION_GROUPS)):
        current = EgoTemporalMoE(**model_args).to(device)
        current.load_state_dict(global_ema.state_dict())
        current_ema = clone_frozen(current).to(device)
        previous_global = clone_frozen(global_ema).to(device) if task > 0 else None
        dataset = dataset_for(
            train_pairs,
            paths,
            exo_pools,
            task,
            "train",
            scores,
            float(cfg.get("synthetic_pair_ratio", 0.30)),
            args.seed,
        )
        clean_dataset = dataset_for(
            train_pairs, paths, exo_pools, task, "train", scores, 0.0, args.seed
        )
        prototype_loader = make_loader(
            clean_dataset, args.batch_size, args.num_workers, False, args.seed + task
        )
        prototypes[task] = task_prototype(prototype_loader, device)
        optimizer = torch.optim.AdamW(
            [
                {"params": current.parameters(), "lr": specialist_lr},
                {
                    "params": global_model.parameters(),
                    "lr": global_lr,
                },
            ],
            weight_decay=effective_weight_decay,
        )
        last_details: Dict[str, float] = {}
        for epoch in range(epochs[task]):
            current.train()
            global_model.train()
            totals: Dict[str, float] = {}
            count = 0
            loader = make_loader(
                dataset,
                args.batch_size,
                args.num_workers,
                True,
                args.seed + task * 1009 + epoch,
            )
            for raw in loader:
                batch = move_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                specialist_loss, specialist_details = train_loss(
                    current, batch, float(cfg.get("token_dropout", 0.08)),
                    float(cfg.get("noise_std", 0.008)),
                    float(cfg.get("consistency_weight", 0.10)),
                )
                global_loss, _ = train_loss(
                    global_model, batch, float(cfg.get("token_dropout", 0.08)),
                    float(cfg.get("noise_std", 0.008)), 0.0,
                )
                loss = specialist_loss + float(cfg.get("global_weight", 0.65)) * global_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*current.parameters(), *global_model.parameters()], 5.0)
                optimizer.step()
                loss_value = float(loss.detach())
                update_ema(current_ema, current, ema_decay)
                update_ema(global_ema, global_model, ema_decay)
                totals["loss"] = totals.get("loss", 0.0) + loss_value
                for key, value in specialist_details.items():
                    totals[key] = totals.get(key, 0.0) + value
                count += 1

            # A bounded rehearsal pass keeps the global expert useful for early
            # sessions without changing any already-frozen task specialist.
            if task > 0:
                replay_budget = int(cfg.get("replay_batches_per_task", 8))
                for old_task in range(task):
                    replay_dataset = dataset_for(
                        train_pairs,
                        paths,
                        exo_pools,
                        old_task,
                        "train",
                        scores,
                        0.0,
                        args.seed,
                    )
                    replay_loader = make_loader(
                        replay_dataset,
                        args.batch_size,
                        args.num_workers,
                        True,
                        args.seed + 20000 + task * 100 + old_task * 10 + epoch,
                    )
                    for replay_index, raw in enumerate(replay_loader):
                        if replay_index >= replay_budget:
                            break
                        batch = move_batch(raw, device)
                        def replay_closure():
                            result = global_model.forward_pair(batch["better"], batch["worse"])
                            replay_loss, _ = ranking_objective(result, batch["skill_gap_target"])
                            with torch.no_grad():
                                teacher = previous_global.forward_pair(batch["better"], batch["worse"])["margin"]
                            distill = F.smooth_l1_loss(result["margin"], teacher)
                            return float(cfg.get("replay_weight", 0.55)) * replay_loss + float(cfg.get("distill_weight", 0.35)) * distill

                        optimizer.zero_grad(set_to_none=True)
                        loss = replay_closure()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(global_model.parameters(), 5.0)
                        optimizer.step()
                        update_ema(global_ema, global_model, ema_decay)
            last_details = {key: value / max(count, 1) for key, value in totals.items()}
            print(
                f"[train] task={task + 1} epoch={epoch + 1}/{epochs[task]} "
                f"loss={last_details.get('loss', float('nan')):.5f}",
                flush=True,
            )

        current_ema.eval().requires_grad_(False)
        experts.append(current_ema)
        global_snapshot = clone_frozen(global_ema).to(device)
        evaluate_session(
            experts,
            global_snapshot,
            prototypes,
            val_pairs,
            paths,
            exo_pools,
            output,
            task + 1,
            device,
            args.batch_size,
            args.num_workers,
        )
        training_log.append({"task": task + 1, "epochs": epochs[task], **last_details})
        atomic_save(
            {
                "method": "flygcl_ego_temporal_moe_v1",
                "seed": args.seed,
                "experts": [model.state_dict() for model in experts],
                "global_model": global_model.state_dict(),
                "global_ema": global_ema.state_dict(),
                "prototypes": prototypes,
                "training_log": training_log,
                "config": config,
                "training_overrides": training_overrides,
            },
            checkpoint,
        )

    payload = {
        "status": "complete",
        "method": "FlyGCL Ego-only task MoE plus replayed global expert and EMA",
        "seed": args.seed,
        "epochs_by_task": epochs,
        "training_log": training_log,
        "prediction_root": str(output),
        "training_overrides": training_overrides,
    }
    final_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
