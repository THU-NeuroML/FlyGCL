#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from peft import PeftModel

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import IMAGENET_STATS, make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.scripts.clare import PeftWrapperPolicy


METHOD_CONFIG = {
    "flyvla_rp": {
        "output_root": "outputs/flyvla_rp",
        "run_name": "flyvla_dit_flow_mt_seed_{seed}_libero_10_task_9_rp10000_r10000_ema0.9p0.99_flyvla_rp",
    },
    "dit_dec": {
        "output_root": "outputs/dit_dec",
        "run_name": "dit_flow_mt_cl_seed_{seed}_libero_10_task_9_encoder_mlp_adapter_threshold_2_5_reproduce",
    },
}


def resolve_dataset_root(dataset_root: str | None, repo_id: str) -> str:
    if dataset_root:
        return dataset_root
    return str(Path(os.environ.get("HF_LEROBOT_HOME", "./.cache/huggingface/lerobot")) / repo_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze final router accuracy for CLARE-style PEFT checkpoints.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--methods", nargs="+", default=["flyvla_rp", "dit_dec"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--tasks", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/router_analysis"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def get_final_paths(repo_root: Path, method: str, seed: int) -> tuple[Path, Path, Path]:
    method_cfg = METHOD_CONFIG[method]
    run_dir = repo_root / method_cfg["output_root"] / f"seed_{seed}" / method_cfg["run_name"].format(seed=seed)
    adapter_dir = run_dir / "checkpoints" / "last" / "adapter"
    pretrained_dir = run_dir / "checkpoints" / "last" / "pretrained_model"
    eval_path = run_dir / "multitask_eval_info.json"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Missing adapter dir: {adapter_dir}")
    if not pretrained_dir.is_dir():
        raise FileNotFoundError(f"Missing pretrained dir: {pretrained_dir}")
    if not eval_path.is_file():
        raise FileNotFoundError(f"Missing eval payload: {eval_path}")
    return adapter_dir, pretrained_dir, eval_path


def build_policy(pretrained_dir: Path, dataset_repo_id: str, dataset_root: str | None, dataset_revision: str | None):
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_dir)
    policy_cfg.pretrained_path = str(pretrained_dir)
    ds_meta = LeRobotDatasetMetadata(
        dataset_repo_id,
        root=resolve_dataset_root(dataset_root, dataset_repo_id),
        revision=dataset_revision,
    )
    for key in ds_meta.camera_keys:
        for stats_type, stats in IMAGENET_STATS.items():
            ds_meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    return policy, policy_cfg


def build_task_dataset(
    base_dataset_cfg: dict,
    policy_cfg: PreTrainedConfig,
    task_id: int,
):
    dataset_cfg = json.loads(json.dumps(base_dataset_cfg))
    dataset_cfg["repo_id"] = f"continuallearning/libero_10_image_task_{task_id}"
    dataset_cfg["root"] = resolve_dataset_root(dataset_cfg.get("root"), dataset_cfg["repo_id"])
    cfg = SimpleNamespace(
        dataset=to_namespace(dataset_cfg),
        policy=policy_cfg,
    )
    dataset = make_dataset(cfg)
    return dataset


def get_predicted_task_ids(peft_module, layer_index: int) -> list[int]:
    info_dicts = peft_module.info_dicts
    selected = info_dicts["top_1_idx_list"]
    if peft_module.routing_mode in {"rp_gate", "ws_router", "whitened_subspace"}:
        adapter_ids = selected
    else:
        adapter_ids = [peft_module.get_adapter_id_by_discriminator_id(discriminator_id) for discriminator_id in selected]

    adapters = peft_module.clare_func_adapters[peft_module.adapter_name]
    return [int(adapters[adapter_id].task_id.item()) for adapter_id in adapter_ids]


def evaluate_task_router(dataset, policy, peft_module, gt_task_id: int, batch_size: int, num_workers: int) -> dict:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(policy.config.device).startswith("cuda"),
        drop_last=False,
    )

    frame_total = 0
    frame_correct = 0
    confusion = Counter()
    episode_votes: dict[int, Counter] = defaultdict(Counter)
    episode_totals = Counter()

    device = next(policy.parameters()).device
    policy.eval()

    for batch in dataloader:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=device.type == "cuda")

        with torch.no_grad():
            policy.forward(batch)

        predicted_task_ids = get_predicted_task_ids(peft_module, 0)
        episode_indices = batch["episode_index"].detach().cpu().tolist()

        for predicted_task_id, episode_index in zip(predicted_task_ids, episode_indices, strict=True):
            frame_total += 1
            frame_correct += int(predicted_task_id == gt_task_id)
            confusion[predicted_task_id] += 1
            episode_votes[episode_index][predicted_task_id] += 1
            episode_totals[episode_index] += 1

    episode_majority_correct = 0
    for episode_index, votes in episode_votes.items():
        majority_task_id, _ = max(votes.items(), key=lambda item: (item[1], -item[0]))
        episode_majority_correct += int(majority_task_id == gt_task_id)

    wrong_confusions = {task_id: count for task_id, count in confusion.items() if task_id != gt_task_id}
    if wrong_confusions:
        top_confused_task, top_confused_count = max(wrong_confusions.items(), key=lambda item: item[1])
        top_confused_share = top_confused_count / frame_total * 100.0
    else:
        top_confused_task, top_confused_share = None, 0.0

    return {
        "frame_acc": frame_correct / frame_total * 100.0,
        "episode_majority_acc": episode_majority_correct / len(episode_votes) * 100.0,
        "frame_total": frame_total,
        "episode_total": len(episode_votes),
        "top_confused_task": top_confused_task,
        "top_confused_share": top_confused_share,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def aggregate_results(raw_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["method"], row["task"])].append(row)

    aggregated = []
    for (method, task), rows in sorted(grouped.items()):
        frame_accs = [row["frame_acc"] for row in rows]
        episode_accs = [row["episode_majority_acc"] for row in rows]
        successes = [row["success"] for row in rows]
        confusion_counter = Counter()
        for row in rows:
            if row["top_confused_task"] is not None:
                confusion_counter[row["top_confused_task"]] += row["top_confused_share"]

        top_confused_task = None
        top_confused_share = 0.0
        if confusion_counter:
            top_confused_task, top_confused_share = confusion_counter.most_common(1)[0]
            top_confused_share /= len(rows)

        aggregated.append(
            {
                "method": method,
                "task": task,
                "frame_acc_mean": mean(frame_accs),
                "frame_acc_std": std(frame_accs),
                "episode_majority_acc_mean": mean(episode_accs),
                "episode_majority_acc_std": std(episode_accs),
                "success_mean": mean(successes),
                "success_std": std(successes),
                "top_confused_task": top_confused_task,
                "top_confused_share_mean": top_confused_share,
            }
        )

    return aggregated


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    raw_rows = []

    for method in args.methods:
        for seed in args.seeds:
            adapter_dir, pretrained_dir, eval_path = get_final_paths(args.repo_root, method, seed)
            train_cfg = load_json(pretrained_dir / "train_config.json")
            final_eval = load_json(eval_path)

            policy, policy_cfg = build_policy(
                pretrained_dir=pretrained_dir,
                dataset_repo_id=train_cfg["dataset"]["repo_id"],
                dataset_root=train_cfg["dataset"]["root"],
                dataset_revision=train_cfg["dataset"]["revision"],
            )
            peft_policy = PeftModel.from_pretrained(
                PeftWrapperPolicy(policy=policy),
                str(adapter_dir),
                is_trainable=False,
                autocast_adapter_dtype=False,
            )
            peft_module = peft_policy.base_model.adapter_layers[args.layer_index]

            for task_id in args.tasks:
                print(f"[router-analysis] method={method} seed={seed} task={task_id}", flush=True)
                dataset = build_task_dataset(train_cfg["dataset"], policy_cfg, task_id)
                metrics = evaluate_task_router(
                    dataset=dataset,
                    policy=policy,
                    peft_module=peft_module,
                    gt_task_id=task_id,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                metrics.update(
                    {
                        "method": method,
                        "seed": seed,
                        "task": task_id,
                        "success": final_eval["per_task"][f"Libero_10_Task_{task_id}"]["pc_success"],
                    }
                )
                raw_rows.append(metrics)

            del peft_policy
            del policy
            torch.cuda.empty_cache()

    aggregated_rows = aggregate_results(raw_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "router_accuracy_raw.json"
    agg_path = args.output_dir / "router_accuracy_summary.json"
    csv_path = args.output_dir / "router_accuracy_summary.csv"

    with open(raw_path, "w") as handle:
        json.dump(raw_rows, handle, indent=2)
    with open(agg_path, "w") as handle:
        json.dump(aggregated_rows, handle, indent=2)

    save_csv(
        csv_path,
        aggregated_rows,
        fieldnames=[
            "method",
            "task",
            "frame_acc_mean",
            "frame_acc_std",
            "episode_majority_acc_mean",
            "episode_majority_acc_std",
            "success_mean",
            "success_std",
            "top_confused_task",
            "top_confused_share_mean",
        ],
    )

    print(f"Saved raw results to {raw_path}")
    print(f"Saved summary JSON to {agg_path}")
    print(f"Saved summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
