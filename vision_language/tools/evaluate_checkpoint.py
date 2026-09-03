#!/usr/bin/env python3
"""Load a final FlyGCL checkpoint and recompute final exposed-class accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from omegaconf import OmegaConf

from continual_clip.OnlineIterDataset import OnlineIterDataset
from continual_clip.analysis.metrics.level1_recompute import load_method_model
from continual_clip.datasets import get_dataset_for_gcl
from continual_clip.utils.onlinesampler import OnlineSampler, OnlineTestSampler
from main_gcl import evaluate_gcl_detailed, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    cfg.dataset_root = str(Path(args.data_root).resolve())
    cfg.workdir = str(Path(__file__).resolve().parents[1])
    seed_everything(int(cfg.seed))

    import clip

    _, transform = clip.load(cfg.model_name, device=args.device, jit=False)
    train, class_names = get_dataset_for_gcl(cfg, True, transform)
    test, _ = get_dataset_for_gcl(cfg, False, transform)
    online_train = OnlineIterDataset(train, 1)
    online_test = OnlineIterDataset(test, 1)
    sampler = OnlineSampler(
        online_train,
        num_tasks=int(cfg.gcl_sessions),
        m=int(cfg.gcl_blurry_ratio),
        n=int(cfg.gcl_disjoint_ratio),
        rnd_seed=int(getattr(cfg, "stream_seed", cfg.seed)),
    )
    plan = []
    for sid in range(int(cfg.gcl_sessions)):
        plan.append(sorted(set(map(int, sampler.disjoint_classes[sid] + sampler.blurry_classes[sid]))))

    model = load_method_model(
        cfg, class_names, plan, args.device, checkpoint, int(cfg.gcl_sessions) - 1
    )
    exposed = sorted({class_id for session in plan for class_id in session})
    loader = torch.utils.data.DataLoader(
        online_test,
        batch_size=int(cfg.batch_size),
        sampler=OnlineTestSampler(online_test, exposed_class=exposed),
        num_workers=int(cfg.num_workers),
        pin_memory=args.device.startswith("cuda"),
    )
    accuracy, per_class = evaluate_gcl_detailed(
        model, loader, exposed, len(class_names), torch.device(args.device)
    )
    result = {
        "dataset": str(cfg.dataset),
        "seed": int(getattr(cfg, "stream_seed", cfg.seed)),
        "checkpoint": str(checkpoint),
        "A_last": accuracy,
        "classes_evaluated": len(exposed),
        "per_class_accuracy": per_class.tolist(),
        "note": "A_auc/F/BWT require the full session trajectory and are not reconstructed from a final checkpoint.",
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
