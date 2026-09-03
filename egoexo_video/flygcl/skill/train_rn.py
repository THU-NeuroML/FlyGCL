#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
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
from flygcl.common.data import ExoPools, SkillDataset, collate_records, fit_bradley_terry_scores, read_pairs
from flygcl.anchor.trainer import move_batch, seed_everything
from flygcl.skill.rn_model import RNTemporalMoE
from flygcl.skill.ego_model import ranking_objective
from flygcl.skill.ego_model import EgoTemporalMoE
from flygcl.skill.utilities import load_prediction_rows


def atomic_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_records,
        generator=torch.Generator().manual_seed(seed),
    )


def augment(value: torch.Tensor, dropout: float, noise: float) -> torch.Tensor:
    keep = (torch.rand(value.shape[:-1] + (1,), device=value.device) >= dropout).to(value.dtype)
    return value * keep + noise * torch.randn_like(value)


@torch.no_grad()
def update_ema(target: RNTemporalMoE, source: RNTemporalMoE, decay: float) -> None:
    target_state, source_state = target.state_dict(), source.state_dict()
    for key, target_value in target_state.items():
        source_value = source_state[key]
        if torch.is_floating_point(target_value):
            target_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
        else:
            target_value.copy_(source_value)


def prediction_path(root: Path, session: int, task: int) -> Path:
    return root / f"task_{session:02d}/predictions/eval_task_{task:02d}.jsonl"


@torch.no_grad()
def evaluate_session(
    experts,
    val_pairs,
    paths,
    pools,
    ego_run: Path,
    output: Path,
    session: int,
    device,
    batch_size: int,
    workers: int,
):
    for model in experts:
        model.eval()
    for task in range(session):
        dataset = SkillDataset(
            val_pairs,
            paths["ego_root"],
            pools,
            "val",
            task,
            "ego_exo",
            exo_references=4,
        )
        route_rows = load_prediction_rows(prediction_path(ego_run, session, task + 1))
        rows = []
        for raw in loader(dataset, batch_size, workers, False, 10000 + session * 10 + task):
            sample_ids = list(raw["sample_id"])
            routes = torch.tensor(
                [route_rows[item]["routed_task"] for item in sample_ids],
                dtype=torch.long,
                device=device,
            )
            batch = move_batch(raw, device)
            margins = torch.zeros(len(sample_ids), device=device)
            consensus = torch.zeros(len(sample_ids), device=device)
            for route in routes.unique(sorted=True):
                index = torch.nonzero(routes == route, as_tuple=False).flatten()
                result = experts[int(route)].forward_pair(
                    batch["better"].index_select(0, index),
                    batch["worse"].index_select(0, index),
                    batch["exo"].index_select(0, index),
                )
                margins.index_copy_(0, index, result["margin"])
                consensus.index_copy_(0, index, result["expert_margins"].mean(-1))
            rows.extend(
                {
                    "sample_id": sample_id,
                    "y_true": 1,
                    "routed_task": int(route),
                    "margin": float(margin),
                    "expert_consensus_margin": float(auxiliary),
                }
                for sample_id, route, margin, auxiliary in zip(
                    sample_ids, routes.cpu(), margins.cpu(), consensus.cpu()
                )
            )
        destination = prediction_path(output, session, task + 1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="FlyGCL 20-token RN residual expert")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ego-checkpoint", required=True)
    parser.add_argument("--ego-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", nargs=4, type=int, default=(12, 14, 9, 6))
    parser.add_argument("--learning-rate", type=float, default=0.00012)
    parser.add_argument("--relation-weight", type=float, default=0.35)
    parser.add_argument("--ema-decay", type=float, default=0.985)
    parser.add_argument("--pair-correction-weight", type=float, default=0.0)
    parser.add_argument("--hard-focus-weight", type=float, default=0.0)
    parser.add_argument("--hard-focus-tau", type=float, default=0.75)
    parser.add_argument("--auto-resume", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError(f"Output must remain inside {ROOT}: {output}")
    output.mkdir(parents=True, exist_ok=True)
    final = output / "training_complete.json"
    if args.auto_resume and final.is_file() and json.loads(final.read_text()).get("status") == "complete":
        print(f"[skip] completed: {output}")
        return 0

    config = load_config(args.config)
    paths = resolve_data_paths(config)
    train_pairs, val_pairs = read_pairs(paths["train_pairs"]), read_pairs(paths["val_pairs"])
    pools = ExoPools(paths["exo_root"], seed=42, val_ratio=0.2)
    scores = fit_bradley_terry_scores(train_pairs, iterations=500)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ego_state = torch.load(args.ego_checkpoint, map_location="cpu", weights_only=False)
    if ego_state.get("method") != "flygcl_ego_temporal_moe_v1":
        raise RuntimeError("Expected the frozen enhanced Ego-only checkpoint")
    experts = []
    training_log = []
    checkpoint = output / "checkpoint.pt"
    if args.auto_resume and checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("method") == "flygcl_rn_temporal_residual_v1":
            for expert_state in state["experts"]:
                model = RNTemporalMoE(
                    relation_weight=args.relation_weight,
                    pair_correction_weight=args.pair_correction_weight,
                ).to(device)
                model.load_state_dict(expert_state)
                model.eval().requires_grad_(False)
                experts.append(model)
            training_log = list(state.get("training_log", []))
            print(f"[resume] restored {len(experts)} RN experts", flush=True)

    for task in range(len(experts), 4):
        model = RNTemporalMoE(
            relation_weight=args.relation_weight,
            pair_correction_weight=args.pair_correction_weight,
        ).to(device)
        model.initialize_from_ego(ego_state["experts"][task])
        ema_model = deepcopy(model).eval().requires_grad_(False)
        ego_teacher = EgoTemporalMoE().to(device)
        ego_teacher.load_state_dict(ego_state["experts"][task])
        ego_teacher.eval().requires_grad_(False)
        dataset = SkillDataset(
            train_pairs,
            paths["ego_root"],
            pools,
            "train",
            task,
            "ego_exo",
            exo_references=4,
            skill_scores=scores,
            synthetic_pair_ratio=0.25,
            synthetic_pair_seed=args.seed,
            synthetic_minimum_gap=0.70,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(args.learning_rate), weight_decay=5e-4
        )
        last_details: Dict[str, float] = {}
        for epoch in range(args.epochs[task]):
            model.train()
            totals: Dict[str, float] = {}
            count = 0
            for raw in loader(dataset, args.batch_size, args.num_workers, True, args.seed + task * 1009 + epoch):
                batch = move_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                result = model.forward_pair(
                    augment(batch["better"], 0.07, 0.007),
                    augment(batch["worse"], 0.07, 0.007),
                    augment(batch["exo"], 0.06, 0.006),
                )
                loss, details = ranking_objective(result, batch["skill_gap_target"])
                if float(args.hard_focus_weight) > 0:
                    with torch.no_grad():
                        ego_margin = ego_teacher.forward_pair(
                            batch["better"], batch["worse"]
                        )["margin"]
                        hard_weight = torch.exp(
                            -ego_margin.abs() / max(float(args.hard_focus_tau), 1e-6)
                        )
                    hard_loss = (
                        hard_weight * F.softplus(-result["margin"] / 0.40)
                    ).sum() / hard_weight.sum().clamp_min(1e-6)
                    loss = loss + float(args.hard_focus_weight) * hard_loss
                    details["hard_focus"] = float(hard_loss.detach())
                loss = loss + 0.01 * result["reference_entropy"].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite RN loss at task {task + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                update_ema(ema_model, model, float(args.ema_decay))
                for key, value in details.items():
                    totals[key] = totals.get(key, 0.0) + value
                count += 1
            last_details = {key: value / max(count, 1) for key, value in totals.items()}
            print(
                f"[train-rn] task={task + 1} epoch={epoch + 1}/{args.epochs[task]} "
                f"loss={last_details.get('loss', float('nan')):.5f}",
                flush=True,
            )
        ema_model.eval().requires_grad_(False)
        experts.append(ema_model)
        evaluate_session(
            experts,
            val_pairs,
            paths,
            pools,
            Path(args.ego_run),
            output,
            task + 1,
            device,
            args.batch_size,
            args.num_workers,
        )
        training_log.append({"task": task + 1, "epochs": args.epochs[task], **last_details})
        atomic_save(
            {
                "method": "flygcl_rn_temporal_residual_v1",
                "seed": args.seed,
                "experts": [expert.state_dict() for expert in experts],
                "training_log": training_log,
                "relation_weight": float(args.relation_weight),
                "ema_decay": float(args.ema_decay),
                "pair_correction_weight": float(args.pair_correction_weight),
                "hard_focus_weight": float(args.hard_focus_weight),
                "hard_focus_tau": float(args.hard_focus_tau),
            },
            checkpoint,
        )
    payload = {
        "status": "complete",
        "method": "FlyGCL RN 20-token temporal residual",
        "seed": args.seed,
        "epochs_by_task": list(args.epochs),
        "training_log": training_log,
        "relation_weight": float(args.relation_weight),
        "ema_decay": float(args.ema_decay),
        "pair_correction_weight": float(args.pair_correction_weight),
        "hard_focus_weight": float(args.hard_focus_weight),
        "hard_focus_tau": float(args.hard_focus_tau),
    }
    final.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
